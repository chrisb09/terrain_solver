// well-known header includes
#include <hdf5.h>
#include <mpi.h>

// project-specific headers
#include "client.h"

// standard library headers
#include <string_view>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

enum class IoMode {
    ParallelHdf5,
    Rank0Gather,
};

enum class SyncMode {
    None,
    Step,
    Report,
};

enum class Hdf5XferMode {
    Collective,
    Independent,
};

enum class SaveMode {
    Periodic,
    Triangular,
};

struct Config {
    std::string model_path;
    std::string input_hdf5;
    std::string output_hdf5;
    std::string device = "CPU";
    int steps = 100;
    int save_every = 1;
    SaveMode save_mode = SaveMode::Periodic;
    int triangular_scale = 1;
    int chunk_size = 60;
    bool write_surface = false;
    float clamp_epsilon = 1e-8F;
    IoMode io_mode = IoMode::ParallelHdf5;
    SyncMode mpi_sync_mode = SyncMode::None;
    Hdf5XferMode hdf5_xfer_mode = Hdf5XferMode::Collective;
    int rank_grid_x = 0;
    int rank_grid_z = 0;
    bool overwrite_output = false;
    bool print_build_timestamp = false;
    int gpus_per_node = 1;
};

struct Range {
    int begin = 0;
    int end = 0;

    int size() const { return end - begin; }
};

struct Decomposition {
    MPI_Comm cart_comm = MPI_COMM_NULL;
    int world_rank = 0;
    int world_size = 1;
    int cart_rank = 0;
    int coords_z = 0;
    int coords_x = 0;
    int ranks_z = 1;
    int ranks_x = 1;

    int north = MPI_PROC_NULL;
    int south = MPI_PROC_NULL;
    int west = MPI_PROC_NULL;
    int east = MPI_PROC_NULL;

    int nx = 0;
    int nz = 0;
    int chunk_size = 0;
    int chunks_x = 0;
    int chunks_z = 0;

    Range chunk_x;
    Range chunk_z;
    Range cell_x;
    Range cell_z;

    int local_nx = 0;
    int local_nz = 0;
};

static void usage() {
    std::cout
        << "Usage:\n"
        << "  terrain_solver --model-path <path> --input-hdf5 <path> --output-hdf5 <path> [options]\n\n"
        << "Options:\n"
        << "  --device <cpu|gpu>               Device to run the solver on (default: cpu)\n"
        << "  --gpus-per-node <int>            Number of GPUs per node (default: 1)\n"
        << "  --steps <int>                    Number of simulation steps (default: 100)\n"
        << "  --save-every <int>               Save every N steps (default: 1)\n"
        << "  --save-mode <periodic|triangular>\n"
        << "                                   Save schedule (default: periodic)\n"
        << "  --triangular-scale <int>         Scale factor for triangular increments\n"
        << "                                   (>=1, default: 1)\n"
        << "  --chunk-size <int>               Chunk edge size (default: 60)\n"
        << "  --write-surface                  Also write terrain+water snapshots\n"
        << "  --io-mode <parallel_hdf5|rank0_gather>\n"
        << "                                   Output strategy (default: parallel_hdf5)\n"
        << "  --mpi-sync-mode <none|step|report>\n"
        << "                                   Barrier policy (default: none)\n"
        << "  --hdf5-xfer-mode <collective|independent>\n"
        << "                                   Parallel HDF5 transfer mode (default: collective)\n"
        << "  --rank-grid-x <int>              Optional rank grid size in X direction\n"
        << "  --rank-grid-z <int>              Optional rank grid size in Z direction\n"
        << "  --overwrite-output               Overwrite existing output file\n"
        << "  --print-build-timestamp          Print build timestamp and exit\n"
        << "  --help                           Show this help message\n";
}

static void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static int get_env_int(const char* key, int default_value) {
    const char* raw = std::getenv(key);
    if (raw == nullptr || raw[0] == '\0') {
        return default_value;
    }
    try {
        return std::stoi(std::string(raw));
    } catch (...) {
        return default_value;
    }
}

static std::string get_env_string(const char* key, const std::string& default_value = "") {
    const char* raw = std::getenv(key);
    if (raw == nullptr) {
        return default_value;
    }
    return std::string(raw);
}

static std::string sync_mode_to_string(SyncMode mode) {
    if (mode == SyncMode::None) {
        return "none";
    }
    if (mode == SyncMode::Step) {
        return "step";
    }
    return "report";
}

static bool equals_ignore_case(std::string_view a, std::string_view b) {
    return std::equal(a.begin(), a.end(),
                      b.begin(), b.end(),
                      [](char a, char b) {
                          return std::tolower(a) == std::tolower(b);
                      });
}

static std::string parse_device(const std::string& value) {
    if (equals_ignore_case(value, "CPU")) {
        return "CPU";
    }
    if (equals_ignore_case(value, "GPU")) {
        return "GPU";
    }
    throw std::runtime_error("Unsupported device: " + value);
}

static IoMode parse_io_mode(const std::string& value) {
    if (value == "parallel_hdf5") {
        return IoMode::ParallelHdf5;
    }
    if (value == "rank0_gather") {
        return IoMode::Rank0Gather;
    }
    throw std::runtime_error("Unsupported --io-mode: " + value);
}

static SyncMode parse_sync_mode(const std::string& value) {
    if (value == "none") {
        return SyncMode::None;
    }
    if (value == "step") {
        return SyncMode::Step;
    }
    if (value == "report") {
        return SyncMode::Report;
    }
    throw std::runtime_error("Unsupported --mpi-sync-mode: " + value);
}

static Hdf5XferMode parse_hdf5_xfer_mode(const std::string& value) {
    if (value == "collective") {
        return Hdf5XferMode::Collective;
    }
    if (value == "independent") {
        return Hdf5XferMode::Independent;
    }
    throw std::runtime_error("Unsupported --hdf5-xfer-mode: " + value);
}

static SaveMode parse_save_mode(const std::string& value) {
    if (value == "periodic") {
        return SaveMode::Periodic;
    }
    if (value == "triangular") {
        return SaveMode::Triangular;
    }
    throw std::runtime_error("Unsupported --save-mode: " + value);
}

static std::size_t estimate_snapshot_count(const Config& cfg) {
    std::size_t count = 1; // step 0
    if (cfg.steps <= 0) {
        return count;
    }

    if (cfg.save_mode == SaveMode::Periodic) {
        count += static_cast<std::size_t>(cfg.steps / cfg.save_every);
        if ((cfg.steps % cfg.save_every) != 0) {
            count += 1; // forced final save
        }
        return count;
    }

    // Triangular schedule: 1, 3, 6, 10, ... (plus initial step 0).
    int next_save_step = cfg.triangular_scale;
    int k = 2;
    int triangular_hits = 0;
    while (next_save_step <= cfg.steps) {
        ++triangular_hits;
        next_save_step += (k * cfg.triangular_scale);
        ++k;
    }
    count += static_cast<std::size_t>(triangular_hits);
    const int last_triangular = next_save_step - ((k - 1) * cfg.triangular_scale);
    if (last_triangular != cfg.steps) {
        count += 1; // forced final save
    }
    return count;
}

static Config parse_args(int argc, char** argv, int rank, int total_ranks) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--help") {
            usage();
            std::exit(0);
        } else if (arg == "--print-build-timestamp") {
            cfg.print_build_timestamp = true;
        } else if (arg == "--write-surface") {
            cfg.write_surface = true;
        } else if (arg == "--overwrite-output") {
            cfg.overwrite_output = true;
        } else if (arg == "--model-path" ||
                   arg == "--input-hdf5" || arg == "--output-hdf5" ||
                   arg == "--steps" || arg == "--device" ||
                   arg == "--gpus-per-node" ||
                   arg == "--save-every" || arg == "--save-mode" || arg == "--triangular-scale" || arg == "--chunk-size" || arg == "--io-mode" ||
                   arg == "--mpi-sync-mode" || arg == "--hdf5-xfer-mode" || arg == "--rank-grid-x" || arg == "--rank-grid-z" || arg == "--clamp-epsilon") {
            require(i + 1 < argc, "Missing value for argument: " + arg);
            const std::string value(argv[++i]);
            if (arg == "--device") {
                cfg.device = parse_device(value); // validate value
            } else if (arg == "--gpus-per-node") {
                const int gpus = std::stoi(value);
                require(gpus >= 0, "--gpus-per-node must be >= 0.");
                // if --device is GPU, then it should be >= 1
                cfg.gpus_per_node = gpus;
                if (rank == 0) {
                    std::cout << "Using " << cfg.gpus_per_node << " GPUs per node." << std::endl;
                }
                std::cout << "Rank " << rank << " of " << total_ranks << " uses GPU index " << (rank % cfg.gpus_per_node) << std::endl;
            } else if (arg == "--input-hdf5") {
                cfg.input_hdf5 = value;
            } else if (arg == "--output-hdf5") {
                cfg.output_hdf5 = value;
            } else if (arg == "--model-path") {
                cfg.model_path = value;
            } else if (arg == "--steps") {
                cfg.steps = std::stoi(value);
            } else if (arg == "--save-every") {
                cfg.save_every = std::stoi(value);
            } else if (arg == "--save-mode") {
                cfg.save_mode = parse_save_mode(value);
            } else if (arg == "--triangular-scale") {
                cfg.triangular_scale = std::stoi(value);
            } else if (arg == "--chunk-size") {
                cfg.chunk_size = std::stoi(value);
            } else if (arg == "--io-mode") {
                cfg.io_mode = parse_io_mode(value);
            } else if (arg == "--mpi-sync-mode") {
                cfg.mpi_sync_mode = parse_sync_mode(value);
            } else if (arg == "--hdf5-xfer-mode") {
                cfg.hdf5_xfer_mode = parse_hdf5_xfer_mode(value);
            } else if (arg == "--rank-grid-x") {
                cfg.rank_grid_x = std::stoi(value);
            } else if (arg == "--rank-grid-z") {
                cfg.rank_grid_z = std::stoi(value);
            }
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (!cfg.print_build_timestamp) {
        require(!cfg.model_path.empty(), "--model-path is required.");
        require(!cfg.input_hdf5.empty(), "--input-hdf5 is required.");
        require(!cfg.output_hdf5.empty(), "--output-hdf5 is required.");
        require(cfg.steps >= 0, "--steps must be >= 0.");
        require(cfg.save_every > 0, "--save-every must be > 0.");
        require(cfg.triangular_scale >= 1, "--triangular-scale must be >= 1.");
        require(cfg.chunk_size > 0, "--chunk-size must be > 0.");
        require(cfg.rank_grid_x >= 0, "--rank-grid-x must be >= 0.");
        require(cfg.rank_grid_z >= 0, "--rank-grid-z must be >= 0.");
    }
    return cfg;
}

static Range partition_range(int total, int parts, int idx) {
    Range r;
    r.begin = (idx * total) / parts;
    r.end = ((idx + 1) * total) / parts;
    return r;
}

static std::string format_duration(std::chrono::seconds seconds) {
    const auto total = seconds.count();
    const auto hours = total / 3600;
    const auto minutes = (total % 3600) / 60;
    const auto secs = total % 60;
    std::ostringstream oss;
    oss << std::setw(2) << std::setfill('0') << hours << ":"
        << std::setw(2) << std::setfill('0') << minutes << ":"
        << std::setw(2) << std::setfill('0') << secs;
    return oss.str();
}

static void check_h5(herr_t code, const std::string& what) {
    if (code < 0) {
        throw std::runtime_error("HDF5 error: " + what);
    }
}

static hid_t create_parallel_fapl(MPI_Comm comm) {
    const hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    require(fapl >= 0, "Failed to create HDF5 file access property list.");
#ifdef H5_HAVE_PARALLEL
    check_h5(H5Pset_fapl_mpio(fapl, comm, MPI_INFO_NULL), "H5Pset_fapl_mpio");
    // Force collective metadata operations (reads and writes) to prevent B-tree
    // corruption when multiple ranks write to chunked datasets concurrently.
    // Data transfers can still be independent; this only affects metadata I/O.
    check_h5(H5Pset_coll_metadata_write(fapl, true), "H5Pset_coll_metadata_write");
    check_h5(H5Pset_all_coll_metadata_ops(fapl, true), "H5Pset_all_coll_metadata_ops");
#else
    H5Pclose(fapl);
    throw std::runtime_error("Parallel HDF5 is not available in this HDF5 build.");
#endif
    return fapl;
}

static hid_t create_collective_dxpl() {
    const hid_t dxpl = H5Pcreate(H5P_DATASET_XFER);
    require(dxpl >= 0, "Failed to create HDF5 transfer property list.");
#ifdef H5_HAVE_PARALLEL
    check_h5(H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_COLLECTIVE), "H5Pset_dxpl_mpio");
#else
    H5Pclose(dxpl);
    throw std::runtime_error("Parallel HDF5 is not available in this HDF5 build.");
#endif
    return dxpl;
}

static hid_t create_independent_dxpl() {
    const hid_t dxpl = H5Pcreate(H5P_DATASET_XFER);
    require(dxpl >= 0, "Failed to create HDF5 transfer property list.");
#ifdef H5_HAVE_PARALLEL
    check_h5(H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_INDEPENDENT), "H5Pset_dxpl_mpio independent");
#else
    H5Pclose(dxpl);
    throw std::runtime_error("Parallel HDF5 is not available in this HDF5 build.");
#endif
    return dxpl;
}

// Serial variant: rank 0 opens with the default (serial) FAPL and broadcasts dims.
static void read_global_dims_serial(const std::string& input_hdf5, std::size_t& nz, std::size_t& nx, int world_rank, MPI_Comm comm) {
    if (world_rank == 0) {
        const hid_t file = H5Fopen(input_hdf5.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        require(file >= 0, "Failed to open input HDF5 file: " + input_hdf5);

        const hid_t terrain_ds = H5Dopen2(file, "terrain", H5P_DEFAULT);
        require(terrain_ds >= 0, "Failed to open dataset 'terrain'.");
        const hid_t terrain_space = H5Dget_space(terrain_ds);
        require(terrain_space >= 0, "Failed to get dataspace for 'terrain'.");

        require(H5Sget_simple_extent_ndims(terrain_space) == 2, "Dataset 'terrain' must be 2D.");
        hsize_t dims[2] = {0, 0};
        check_h5(H5Sget_simple_extent_dims(terrain_space, dims, nullptr), "H5Sget_simple_extent_dims terrain");
        nz = static_cast<std::size_t>(dims[0]);
        nx = static_cast<std::size_t>(dims[1]);

        H5Sclose(terrain_space);
        H5Dclose(terrain_ds);
        H5Fclose(file);
    }
    // Broadcast to all ranks.
    hsize_t buf[2] = {static_cast<hsize_t>(nz), static_cast<hsize_t>(nx)};
    MPI_Bcast(buf, 2, MPI_UNSIGNED_LONG_LONG, 0, comm);
    nz = static_cast<std::size_t>(buf[0]);
    nx = static_cast<std::size_t>(buf[1]);
}

static void read_global_dims_parallel(const std::string& input_hdf5, std::size_t& nz, std::size_t& nx, MPI_Comm comm) {
    const hid_t fapl = create_parallel_fapl(comm);
    const hid_t file = H5Fopen(input_hdf5.c_str(), H5F_ACC_RDONLY, fapl);
    H5Pclose(fapl);
    require(file >= 0, "Failed to open input HDF5 file: " + input_hdf5);

    const hid_t terrain_ds = H5Dopen2(file, "terrain", H5P_DEFAULT);
    require(terrain_ds >= 0, "Failed to open dataset 'terrain'.");
    const hid_t terrain_space = H5Dget_space(terrain_ds);
    require(terrain_space >= 0, "Failed to get dataspace for 'terrain'.");

    require(H5Sget_simple_extent_ndims(terrain_space) == 2, "Dataset 'terrain' must be 2D.");
    hsize_t dims[2] = {0, 0};
    check_h5(H5Sget_simple_extent_dims(terrain_space, dims, nullptr), "H5Sget_simple_extent_dims terrain");
    nz = static_cast<std::size_t>(dims[0]);
    nx = static_cast<std::size_t>(dims[1]);

    H5Sclose(terrain_space);
    H5Dclose(terrain_ds);
    H5Fclose(file);
}

// Serial variant: rank 0 reads the full global dataset and scatters local slices.
static std::vector<float> read_local_2d_rank0_gather(
    const std::string& input_hdf5,
    const char* dataset_name,
    const Decomposition& decomp,
    std::size_t expected_nz,
    std::size_t expected_nx) {

    std::vector<float> send_buf;
    std::vector<int> counts(static_cast<std::size_t>(decomp.world_size), 0);
    std::vector<int> displs(static_cast<std::size_t>(decomp.world_size), 0);

    if (decomp.world_rank == 0) {
        // Read entire global field.
        const hid_t file = H5Fopen(input_hdf5.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        require(file >= 0, "Failed to open input HDF5 file: " + input_hdf5);
        const hid_t ds = H5Dopen2(file, dataset_name, H5P_DEFAULT);
        require(ds >= 0, std::string("Failed to open dataset '") + dataset_name + "'.");
        const hid_t fspace = H5Dget_space(ds);
        require(fspace >= 0, "Failed to get input dataset dataspace.");
        require(H5Sget_simple_extent_ndims(fspace) == 2, std::string("Dataset must be 2D: ") + dataset_name);
        hsize_t dims[2] = {0, 0};
        check_h5(H5Sget_simple_extent_dims(fspace, dims, nullptr), "H5Sget_simple_extent_dims input serial");
        require(static_cast<std::size_t>(dims[0]) == expected_nz && static_cast<std::size_t>(dims[1]) == expected_nx,
                std::string("Unexpected dimensions in dataset ") + dataset_name);
        std::vector<float> global(expected_nz * expected_nx);
        check_h5(H5Dread(ds, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, global.data()),
                 std::string("H5Dread serial ") + dataset_name);
        H5Sclose(fspace);
        H5Dclose(ds);
        H5Fclose(file);

        // Compute per-rank send counts.
        for (int r = 0; r < decomp.world_size; ++r) {
            int coords[2] = {0, 0};
            MPI_Cart_coords(decomp.cart_comm, r, 2, coords);
            const Range rz = partition_range(decomp.chunks_z, decomp.ranks_z, coords[0]);
            const Range rx = partition_range(decomp.chunks_x, decomp.ranks_x, coords[1]);
            counts[static_cast<std::size_t>(r)] = (rz.end - rz.begin) * decomp.chunk_size
                                                 * (rx.end - rx.begin) * decomp.chunk_size;
        }
        int offset = 0;
        for (int r = 0; r < decomp.world_size; ++r) {
            displs[static_cast<std::size_t>(r)] = offset;
            offset += counts[static_cast<std::size_t>(r)];
        }
        send_buf.resize(static_cast<std::size_t>(offset));

        // Pack each rank's local region (row-major) into send_buf.
        for (int r = 0; r < decomp.world_size; ++r) {
            int coords[2] = {0, 0};
            MPI_Cart_coords(decomp.cart_comm, r, 2, coords);
            const Range rz = partition_range(decomp.chunks_z, decomp.ranks_z, coords[0]);
            const Range rx = partition_range(decomp.chunks_x, decomp.ranks_x, coords[1]);
            const int z0 = rz.begin * decomp.chunk_size;
            const int lnz = (rz.end - rz.begin) * decomp.chunk_size;
            const int x0 = rx.begin * decomp.chunk_size;
            const int lnx = (rx.end - rx.begin) * decomp.chunk_size;
            float* dst = send_buf.data() + displs[static_cast<std::size_t>(r)];
            for (int z = 0; z < lnz; ++z) {
                const float* src = global.data() + static_cast<std::size_t>(z0 + z) * expected_nx
                                   + static_cast<std::size_t>(x0);
                std::copy(src, src + lnx, dst + z * lnx);
            }
        }
    }

    const int local_count = decomp.local_nz * decomp.local_nx;
    std::vector<float> local(static_cast<std::size_t>(local_count));
    MPI_Scatterv(
        send_buf.data(),
        counts.data(),
        displs.data(),
        MPI_FLOAT,
        local.data(),
        local_count,
        MPI_FLOAT,
        0,
        decomp.cart_comm);
    return local;
}

static std::vector<float> read_local_2d_parallel(
    const std::string& input_hdf5,
    const char* dataset_name,
    const Decomposition& decomp,
    std::size_t expected_nz,
    std::size_t expected_nx) {
    const hid_t fapl = create_parallel_fapl(decomp.cart_comm);
    const hid_t file = H5Fopen(input_hdf5.c_str(), H5F_ACC_RDONLY, fapl);
    H5Pclose(fapl);
    require(file >= 0, "Failed to open input HDF5 file: " + input_hdf5);

    const hid_t ds = H5Dopen2(file, dataset_name, H5P_DEFAULT);
    require(ds >= 0, std::string("Failed to open dataset '") + dataset_name + "'.");
    const hid_t file_space = H5Dget_space(ds);
    require(file_space >= 0, "Failed to get input dataset dataspace.");

    require(H5Sget_simple_extent_ndims(file_space) == 2, std::string("Dataset must be 2D: ") + dataset_name);
    hsize_t dims[2] = {0, 0};
    check_h5(H5Sget_simple_extent_dims(file_space, dims, nullptr), "H5Sget_simple_extent_dims input");
    require(static_cast<std::size_t>(dims[0]) == expected_nz && static_cast<std::size_t>(dims[1]) == expected_nx,
            std::string("Unexpected dimensions in dataset ") + dataset_name);

    const hsize_t start[2] = {
        static_cast<hsize_t>(decomp.cell_z.begin),
        static_cast<hsize_t>(decomp.cell_x.begin),
    };
    const hsize_t count[2] = {
        static_cast<hsize_t>(decomp.local_nz),
        static_cast<hsize_t>(decomp.local_nx),
    };

    check_h5(H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, nullptr, count, nullptr),
             "H5Sselect_hyperslab input");

    const hid_t mem_space = H5Screate_simple(2, count, nullptr);
    require(mem_space >= 0, "Failed to create input memspace.");

    std::vector<float> local(static_cast<std::size_t>(decomp.local_nz) * static_cast<std::size_t>(decomp.local_nx), 0.0F);

    const hid_t dxpl = create_collective_dxpl();
    check_h5(H5Dread(ds, H5T_NATIVE_FLOAT, mem_space, file_space, dxpl, local.data()),
             std::string("H5Dread ") + dataset_name);

    H5Pclose(dxpl);
    H5Sclose(mem_space);
    H5Sclose(file_space);
    H5Dclose(ds);
    H5Fclose(file);

    return local;
}

static std::size_t grid_index(std::size_t z, std::size_t x, std::size_t nx) {
    return z * nx + x;
}

static std::size_t local_index(int i, int j, int pitch) {
    return static_cast<std::size_t>(i) * static_cast<std::size_t>(pitch) + static_cast<std::size_t>(j);
}

static void copy_packed_to_with_halo(const std::vector<float>& packed, std::vector<float>& with_halo, int local_nz, int local_nx) {
    const int pitch = local_nx + 2;
    for (int z = 0; z < local_nz; ++z) {
        const float* src = &packed[static_cast<std::size_t>(z) * static_cast<std::size_t>(local_nx)];
        float* dst = &with_halo[local_index(z + 1, 1, pitch)];
        std::copy(src, src + local_nx, dst);
    }
}

static std::vector<float> pack_interior(const std::vector<float>& with_halo, int local_nz, int local_nx) {
    const int pitch = local_nx + 2;
    std::vector<float> packed(static_cast<std::size_t>(local_nz) * static_cast<std::size_t>(local_nx), 0.0F);
    for (int z = 0; z < local_nz; ++z) {
        const float* src = &with_halo[local_index(z + 1, 1, pitch)];
        float* dst = &packed[static_cast<std::size_t>(z) * static_cast<std::size_t>(local_nx)];
        std::copy(src, src + local_nx, dst);
    }
    return packed;
}

static void clamp_nonnegative_interior(std::vector<float>& field, int local_nz, int local_nx) {
    const int pitch = local_nx + 2;
    for (int i = 1; i <= local_nz; ++i) {
        for (int j = 1; j <= local_nx; ++j) {
            float& v = field[local_index(i, j, pitch)];
            if (v < 0.0F) {
                v = 0.0F;
            }
        }
    }
}

static void exchange_halo_1cell(std::vector<float>& field, const Decomposition& decomp, int tag_base) {
    const int local_nz = decomp.local_nz;
    const int local_nx = decomp.local_nx;
    const int pitch = local_nx + 2;

    std::vector<float> send_west(static_cast<std::size_t>(local_nz));
    std::vector<float> send_east(static_cast<std::size_t>(local_nz));
    std::vector<float> recv_west(static_cast<std::size_t>(local_nz));
    std::vector<float> recv_east(static_cast<std::size_t>(local_nz));

    for (int i = 0; i < local_nz; ++i) {
        send_west[static_cast<std::size_t>(i)] = field[local_index(i + 1, 1, pitch)];
        send_east[static_cast<std::size_t>(i)] = field[local_index(i + 1, local_nx, pitch)];
    }

    enum {
        TAG_TO_NORTH = 0,
        TAG_TO_SOUTH = 1,
        TAG_TO_WEST = 2,
        TAG_TO_EAST = 3,
    };

    std::array<MPI_Request, 8> reqs;
    int nreq = 0;

    MPI_Irecv(&field[local_index(0, 1, pitch)], local_nx, MPI_FLOAT, decomp.north, tag_base + TAG_TO_SOUTH, decomp.cart_comm,
              &reqs[static_cast<std::size_t>(nreq++)]);
    MPI_Irecv(&field[local_index(local_nz + 1, 1, pitch)], local_nx, MPI_FLOAT, decomp.south, tag_base + TAG_TO_NORTH,
              decomp.cart_comm, &reqs[static_cast<std::size_t>(nreq++)]);

    MPI_Irecv(recv_west.data(), local_nz, MPI_FLOAT, decomp.west, tag_base + TAG_TO_EAST, decomp.cart_comm,
              &reqs[static_cast<std::size_t>(nreq++)]);
    MPI_Irecv(recv_east.data(), local_nz, MPI_FLOAT, decomp.east, tag_base + TAG_TO_WEST, decomp.cart_comm,
              &reqs[static_cast<std::size_t>(nreq++)]);

    MPI_Isend(&field[local_index(1, 1, pitch)], local_nx, MPI_FLOAT, decomp.north, tag_base + TAG_TO_NORTH, decomp.cart_comm,
              &reqs[static_cast<std::size_t>(nreq++)]);
    MPI_Isend(&field[local_index(local_nz, 1, pitch)], local_nx, MPI_FLOAT, decomp.south, tag_base + TAG_TO_SOUTH,
              decomp.cart_comm, &reqs[static_cast<std::size_t>(nreq++)]);

    MPI_Isend(send_west.data(), local_nz, MPI_FLOAT, decomp.west, tag_base + TAG_TO_WEST, decomp.cart_comm,
              &reqs[static_cast<std::size_t>(nreq++)]);
    MPI_Isend(send_east.data(), local_nz, MPI_FLOAT, decomp.east, tag_base + TAG_TO_EAST, decomp.cart_comm,
              &reqs[static_cast<std::size_t>(nreq++)]);

    MPI_Waitall(nreq, reqs.data(), MPI_STATUSES_IGNORE);

    for (int i = 0; i < local_nz; ++i) {
        field[local_index(i + 1, 0, pitch)] = recv_west[static_cast<std::size_t>(i)];
        field[local_index(i + 1, local_nx + 1, pitch)] = recv_east[static_cast<std::size_t>(i)];
    }
}

bool hasPrinted = false;
long long total_prepare_data_time = 0;
long long total_put_tensor_time = 0;
long long total_run_model_time = 0;
long long total_unpack_time = 0;
long long total_cleanup_time = 0;
long long total_ml_step_wall_time = 0;

struct MlNestedBatchView {
    std::vector<float***> batch;
    std::vector<float**> channels;
    std::vector<float*> rows;
    const float* base_data = nullptr;
};

struct StepScratch {
    std::vector<float> q_west;
    std::vector<float> q_east;
    std::vector<float> q_north;
    std::vector<float> q_south;
    std::vector<float> outflow_sum;

    StepScratch() = default;

    StepScratch(int local_nz, int local_nx) {
        resize(local_nz, local_nx);
    }

    void resize(int local_nz, int local_nx) {
        const std::size_t cell_count =
            static_cast<std::size_t>(local_nz) * static_cast<std::size_t>(local_nx);
        q_west.resize(cell_count);
        q_east.resize(cell_count);
        q_north.resize(cell_count);
        q_south.resize(cell_count);
        outflow_sum.resize(cell_count);
    }
};

static MlNestedBatchView create_ml_nested_batch_view(const std::vector<float>& field, const Decomposition& decomp) {
    MlNestedBatchView view;
    const int local_nz = decomp.local_nz;
    const int local_nx = decomp.local_nx;
    const int pitch = local_nx + 2;
    const std::size_t batch_size = static_cast<std::size_t>(local_nz) * static_cast<std::size_t>(local_nx);

    view.batch.resize(batch_size);
    view.channels.resize(batch_size);
    view.rows.resize(batch_size * 3);
    view.base_data = field.data();

    for (int i = 0; i < local_nz; ++i) {
        for (int j = 0; j < local_nx; ++j) {
            const int ii = i + 1;
            const int jj = j + 1;
            const std::size_t b = static_cast<std::size_t>(i * local_nx + j);

            view.batch[b] = &view.channels[b];
            view.channels[b] = &view.rows[b * 3];

            for (int di = -1; di <= 1; ++di) {
                const int n_i = ii + di;
                const int n_j_start = jj - 1;
                view.rows[b * 3 + static_cast<std::size_t>(di + 1)] = const_cast<float*>(&field[local_index(n_i, n_j_start, pitch)]);
            }
        }
    }

    return view;
}

static double compute_local_step_ml(SmartRedis::Client* client,
    const std::vector<float>& terrain,
    const std::vector<float>& current,
    std::vector<float>& next,
    const Decomposition& decomp,
    float epsilon,
    const Config& cfg,
    const MlNestedBatchView& batch_input_view,
    std::vector<float>& tile_output) {
        const int local_nz = decomp.local_nz;
        const int local_nx = decomp.local_nx;
        const int pitch = local_nx + 2;

        long long prepare_data_time = 0;
        long long put_tensor_time = 0;
        long long run_model_time = 0;
        long long unpack_time = 0;
        long long cleanup_time = 0;

        next = current;

        const std::size_t BATCH_SIZE = static_cast<std::size_t>(decomp.local_nz) * static_cast<std::size_t>(decomp.local_nx);
        require(batch_input_view.batch.size() == BATCH_SIZE,
                "ML nested batch view size mismatch.");
        require(batch_input_view.base_data == current.data(),
                "ML nested batch view does not match active current buffer.");
        require(tile_output.size() == BATCH_SIZE,
                "ML output buffer size mismatch.");

        auto start = std::chrono::high_resolution_clock::now();

        float**** batch_input = const_cast<float****>(batch_input_view.batch.data());

        const std::string water_key = "water_tile_" + std::to_string(decomp.world_rank);
        const std::string terrain_key = "terrain_tile_" + std::to_string(decomp.world_rank);
        const std::string pred_key = "predicted_water_center_" + std::to_string(decomp.world_rank);

        double moved = 0.0;

        auto data_prepared_time = std::chrono::high_resolution_clock::now();

        client->put_tensor(
            water_key,
            batch_input,
            std::vector<size_t>{BATCH_SIZE, 1, 3, 3},
            SRTensorType::SRTensorTypeFloat,
            SRMemoryLayout::SRMemLayoutNested);

        auto put_time = std::chrono::high_resolution_clock::now();


        if (cfg.device == "GPU") {
            if (cfg.gpus_per_node == 0) {
                throw std::runtime_error("GPU device specified but --gpus-per-node is 0.");
            }
            client->run_model_multigpu("water_step_model_gpu", {water_key, terrain_key}, {pred_key}, decomp.world_rank % cfg.gpus_per_node, 0, cfg.gpus_per_node);
        } else {
            client->run_model("water_step_model", {water_key, terrain_key}, {pred_key});
        }

        auto model_ran_time = std::chrono::high_resolution_clock::now();

        if (decomp.world_rank == 0) {
            std::cout << " and ran model for cell (" << 0 << ", " << 0 << ")" << std::flush;
        }


        std::vector<size_t> large_output_dims = {BATCH_SIZE}; //Batch size of 1 for single cell prediction
        SRTensorType large_output_type = SRTensorType::SRTensorTypeFloat;
        SRMemoryLayout large_output_mem_layout = SRMemoryLayout::SRMemLayoutContiguous;
        
        client->unpack_tensor(pred_key, tile_output.data(), large_output_dims, large_output_type, large_output_mem_layout);
        for (int i = 0; i < decomp.local_nz; ++i) {
            for (int j = 0; j < decomp.local_nx; ++j) {
                next[local_index(i + 1, j + 1, pitch)] = tile_output[static_cast<size_t>(i * local_nx + j)];
                moved += std::max(static_cast<double>(tile_output[static_cast<size_t>(i * local_nx + j)]) - static_cast<double>(current[local_index(i + 1, j + 1, pitch)]), 0.0);
            }
        }
        //next[local_index(ii, jj, pitch)] = tile_output[0];
        //moved += std::max(static_cast<double>(tile_output[0]) - static_cast<double>(current[local_index(ii, jj, pitch)]), 0.0);

        auto unpacked_time = std::chrono::high_resolution_clock::now();

        if (decomp.world_rank == 0) {
            std::cout << " and got prediction " << tile_output[0] << " for cell (" << 0 << ", " << 0 << ")" << std::endl;
        }

        prepare_data_time += std::chrono::duration_cast<std::chrono::microseconds>(data_prepared_time - start).count();
        put_tensor_time += std::chrono::duration_cast<std::chrono::microseconds>(put_time - data_prepared_time).count();
        run_model_time += std::chrono::duration_cast<std::chrono::microseconds>(model_ran_time - put_time).count();
        unpack_time += std::chrono::duration_cast<std::chrono::microseconds>(unpacked_time - model_ran_time).count();
        

        total_prepare_data_time += prepare_data_time;
        total_put_tensor_time += put_tensor_time;
        total_run_model_time += run_model_time;
        total_unpack_time += unpack_time;

        if (decomp.world_rank == 0) {
            std::cout << "Finished local step with ML model. Timings for this step (seconds):" << std::endl;
            std::cout << "Prepare data time (seconds): " << prepare_data_time / 1000000.0 << std::endl;
            std::cout << "Put tensor time (seconds): " << put_tensor_time / 1000000.0 << std::endl;
            std::cout << "Run model time (seconds): " << run_model_time / 1000000.0 << std::endl;
            std::cout << "Unpack time (seconds): " << unpack_time / 1000000.0 << std::endl;


            std::time_t t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            std::tm ltime;
            localtime_r(&t, &ltime);
            std::cout << "Current real-world time (hh:mm:ss): " << std::put_time(&ltime, "%H:%M:%S") << std::endl;

        }

        const auto end = std::chrono::high_resolution_clock::now();
        const long long ml_step_wall_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        total_cleanup_time += cleanup_time;
        total_ml_step_wall_time += ml_step_wall_time;

        if (decomp.world_rank == 0) {
            const long long accounted_us = prepare_data_time + put_tensor_time + run_model_time + unpack_time + cleanup_time;
            std::cout << "ML timing accounting (seconds): total=" << (ml_step_wall_time / 1000000.0)
                      << ", accounted=" << (accounted_us / 1000000.0)
                      << ", cleanup=" << (cleanup_time / 1000000.0)
                      << std::endl;
        }

        return moved;
    }

static float compute_directional_outflow(
    const std::vector<float>& terrain,
    const std::vector<float>& current,
    int i,
    int j,
    int dir,
    int pitch,
    float epsilon) {
    const int rows = static_cast<int>(current.size() / static_cast<std::size_t>(pitch));
    if (i < 0 || i >= rows || j < 0 || j >= pitch) {
        return 0.0F;
    }

    const std::size_t center_idx = local_index(i, j, pitch);
    const float water_here = current[center_idx];
    if (water_here <= epsilon) {
        return 0.0F;
    }

    const float surface_here = terrain[center_idx] + water_here;
    const std::array<std::pair<int, int>, 4> offsets = {
        std::make_pair(0, -1),
        std::make_pair(0, 1),
        std::make_pair(-1, 0),
        std::make_pair(1, 0),
    };

    if (dir < 0 || dir >= 4) {
        return 0.0F;
    }

    std::array<float, 4> q = {0.0F, 0.0F, 0.0F, 0.0F};
    float sum_out = 0.0F;
    for (int k = 0; k < 4; ++k) {
        const int ni = i + offsets[static_cast<std::size_t>(k)].first;
        const int nj = j + offsets[static_cast<std::size_t>(k)].second;
        if (ni < 0 || ni >= rows || nj < 0 || nj >= pitch) {
            continue;
        }

        const std::size_t nidx = local_index(ni, nj, pitch);
        const float surface_neighbor = terrain[nidx] + current[nidx];
        const float raw_diff = surface_here - surface_neighbor;
        if (raw_diff <= epsilon) {
            continue;
        }

        const float effective_diff = std::min(raw_diff, water_here);
        q[static_cast<std::size_t>(k)] = 0.25F * effective_diff;
        sum_out += q[static_cast<std::size_t>(k)];
    }

    if (sum_out <= epsilon) {
        return 0.0F;
    }

    if (sum_out > water_here) {
        const float scale = water_here / sum_out;
        return q[static_cast<std::size_t>(dir)] * scale;
    }
    return q[static_cast<std::size_t>(dir)];
}

static double compute_local_step(
    const std::vector<float>& terrain,
    const std::vector<float>& current,
    std::vector<float>& next,
    const Decomposition& decomp,
    float epsilon,
    StepScratch& scratch) {
    const int local_nz = decomp.local_nz;
    const int local_nx = decomp.local_nx;
    const int pitch = local_nx + 2;
    const std::size_t cell_count =
        static_cast<std::size_t>(local_nz) * static_cast<std::size_t>(local_nx);

    require(scratch.q_west.size() == cell_count &&
                scratch.q_east.size() == cell_count &&
                scratch.q_north.size() == cell_count &&
                scratch.q_south.size() == cell_count &&
                scratch.outflow_sum.size() == cell_count,
            "Step scratch size mismatch.");

    next = current;

    double moved = 0.0;

    for (int i = 1; i <= local_nz; ++i) {
        const std::size_t row_offset = static_cast<std::size_t>(i - 1) * static_cast<std::size_t>(local_nx);
        for (int j = 1; j <= local_nx; ++j) {
            const std::size_t center = local_index(i, j, pitch);
            const std::size_t scratch_idx = row_offset + static_cast<std::size_t>(j - 1);
            const float water_here = current[center];

            float q_west = 0.0F;
            float q_east = 0.0F;
            float q_north = 0.0F;
            float q_south = 0.0F;
            float outflow = 0.0F;

            if (water_here > epsilon) {
                const float surface_here = terrain[center] + water_here;

                const std::size_t west = local_index(i, j - 1, pitch);
                const float west_diff = surface_here - (terrain[west] + current[west]);
                if (west_diff > epsilon) {
                    q_west = 0.25F * std::min(west_diff, water_here);
                    outflow += q_west;
                }

                const std::size_t east = local_index(i, j + 1, pitch);
                const float east_diff = surface_here - (terrain[east] + current[east]);
                if (east_diff > epsilon) {
                    q_east = 0.25F * std::min(east_diff, water_here);
                    outflow += q_east;
                }

                const std::size_t north = local_index(i - 1, j, pitch);
                const float north_diff = surface_here - (terrain[north] + current[north]);
                if (north_diff > epsilon) {
                    q_north = 0.25F * std::min(north_diff, water_here);
                    outflow += q_north;
                }

                const std::size_t south = local_index(i + 1, j, pitch);
                const float south_diff = surface_here - (terrain[south] + current[south]);
                if (south_diff > epsilon) {
                    q_south = 0.25F * std::min(south_diff, water_here);
                    outflow += q_south;
                }

                if (outflow > water_here) {
                    const float scale = water_here / outflow;
                    q_west *= scale;
                    q_east *= scale;
                    q_north *= scale;
                    q_south *= scale;
                    outflow = water_here;
                } else if (outflow <= epsilon) {
                    q_west = 0.0F;
                    q_east = 0.0F;
                    q_north = 0.0F;
                    q_south = 0.0F;
                    outflow = 0.0F;
                }
            }

            scratch.q_west[scratch_idx] = q_west;
            scratch.q_east[scratch_idx] = q_east;
            scratch.q_north[scratch_idx] = q_north;
            scratch.q_south[scratch_idx] = q_south;
            scratch.outflow_sum[scratch_idx] = outflow;
        }
    }

    for (int i = 1; i <= local_nz; ++i) {
        const std::size_t row_offset = static_cast<std::size_t>(i - 1) * static_cast<std::size_t>(local_nx);
        for (int j = 1; j <= local_nx; ++j) {
            const std::size_t center = local_index(i, j, pitch);
            const std::size_t scratch_idx = row_offset + static_cast<std::size_t>(j - 1);
            const float water_here = current[center];
            const float outflow = scratch.outflow_sum[scratch_idx];
            moved += static_cast<double>(outflow);

            const float inflow_from_west =
                (j > 1)
                    ? scratch.q_east[scratch_idx - 1]
                    : compute_directional_outflow(terrain, current, i, j - 1, 1, pitch, epsilon);
            const float inflow_from_east =
                (j < local_nx)
                    ? scratch.q_west[scratch_idx + 1]
                    : compute_directional_outflow(terrain, current, i, j + 1, 0, pitch, epsilon);
            const float inflow_from_north =
                (i > 1)
                    ? scratch.q_south[scratch_idx - static_cast<std::size_t>(local_nx)]
                    : compute_directional_outflow(terrain, current, i - 1, j, 3, pitch, epsilon);
            const float inflow_from_south =
                (i < local_nz)
                    ? scratch.q_north[scratch_idx + static_cast<std::size_t>(local_nx)]
                    : compute_directional_outflow(terrain, current, i + 1, j, 2, pitch, epsilon);
            const float inflow = inflow_from_west + inflow_from_east + inflow_from_north + inflow_from_south;

            next[center] = std::max(0.0F, water_here + inflow - outflow);
        }
    }

    clamp_nonnegative_interior(next, local_nz, local_nx);
    return moved;
}

static std::vector<float> gather_global_field(
    const std::vector<float>& local_packed,
    const Decomposition& decomp,
    MPI_Comm comm,
    int global_nz,
    int global_nx) {
    const int local_count = static_cast<int>(local_packed.size());
    std::vector<int> counts;
    if (decomp.world_rank == 0) {
        counts.resize(static_cast<std::size_t>(decomp.world_size), 0);
    }

    MPI_Gather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, comm);

    std::vector<int> displs;
    int total_count = 0;
    if (decomp.world_rank == 0) {
        displs.resize(static_cast<std::size_t>(decomp.world_size), 0);
        for (int r = 0; r < decomp.world_size; ++r) {
            displs[static_cast<std::size_t>(r)] = total_count;
            total_count += counts[static_cast<std::size_t>(r)];
        }
    }

    std::vector<float> gathered;
    if (decomp.world_rank == 0) {
        gathered.resize(static_cast<std::size_t>(total_count), 0.0F);
    }

    MPI_Gatherv(
        local_packed.data(),
        local_count,
        MPI_FLOAT,
        gathered.data(),
        counts.data(),
        displs.data(),
        MPI_FLOAT,
        0,
        comm);

    if (decomp.world_rank != 0) {
        return {};
    }

    std::vector<float> global(static_cast<std::size_t>(global_nz) * static_cast<std::size_t>(global_nx), 0.0F);

    for (int r = 0; r < decomp.world_size; ++r) {
        int coords[2] = {0, 0};
        MPI_Cart_coords(decomp.cart_comm, r, 2, coords);
        const Range chunk_z = partition_range(decomp.chunks_z, decomp.ranks_z, coords[0]);
        const Range chunk_x = partition_range(decomp.chunks_x, decomp.ranks_x, coords[1]);
        const int z0 = chunk_z.begin * decomp.chunk_size;
        const int z1 = chunk_z.end * decomp.chunk_size;
        const int x0 = chunk_x.begin * decomp.chunk_size;
        const int x1 = chunk_x.end * decomp.chunk_size;
        const int lnz = z1 - z0;
        const int lnx = x1 - x0;

        const int count = counts[static_cast<std::size_t>(r)];
        require(count == lnz * lnx, "Gathered field count mismatch while unpacking rank data.");
        const float* src = gathered.data() + displs[static_cast<std::size_t>(r)];

        for (int z = 0; z < lnz; ++z) {
            float* dst = &global[grid_index(static_cast<std::size_t>(z0 + z), static_cast<std::size_t>(x0), static_cast<std::size_t>(global_nx))];
            std::copy(src + static_cast<std::size_t>(z) * static_cast<std::size_t>(lnx),
                      src + static_cast<std::size_t>(z + 1) * static_cast<std::size_t>(lnx),
                      dst);
        }
    }

    return global;
}

struct SavedStepMetadata {
    int step = 0;
    int solver_type = 0; // 0=init, 1=regular, 2=ml
    double mass = 0.0;
    double drift = 0.0;
    double moved_this_step = 0.0;
    float min_water = 0.0F;
    float min_positive_water = 0.0F;
    float max_water = 0.0F;
    double runtime_seconds = 0.0;
};

class TrajectoryWriter {
public:
    TrajectoryWriter(
        const Config& cfg,
        const Decomposition& decomp,
        const std::vector<float>& terrain_local_packed,
        MPI_Comm comm)
        : cfg_(cfg), decomp_(decomp), comm_(comm), terrain_local_packed_(terrain_local_packed) {
        expected_snapshots_ = estimate_snapshot_count(cfg_);
        setup_file();
        create_datasets_and_write_terrain();
    }

    ~TrajectoryWriter() {
        close();
    }

    void close() {
        if (closed_) {
            return;
        }

        if (ds_step_ >= 0) {
            H5Dclose(ds_step_);
            ds_step_ = -1;
        }
        if (ds_solver_type_ >= 0) {
            H5Dclose(ds_solver_type_);
            ds_solver_type_ = -1;
        }
        if (ds_mass_ >= 0) {
            H5Dclose(ds_mass_);
            ds_mass_ = -1;
        }
        if (ds_drift_ >= 0) {
            H5Dclose(ds_drift_);
            ds_drift_ = -1;
        }
        if (ds_moved_this_step_ >= 0) {
            H5Dclose(ds_moved_this_step_);
            ds_moved_this_step_ = -1;
        }
        if (ds_min_water_ >= 0) {
            H5Dclose(ds_min_water_);
            ds_min_water_ = -1;
        }
        if (ds_min_positive_water_ >= 0) {
            H5Dclose(ds_min_positive_water_);
            ds_min_positive_water_ = -1;
        }
        if (ds_max_water_ >= 0) {
            H5Dclose(ds_max_water_);
            ds_max_water_ = -1;
        }
        if (ds_runtime_seconds_ >= 0) {
            H5Dclose(ds_runtime_seconds_);
            ds_runtime_seconds_ = -1;
        }
        if (ds_surface_ >= 0) {
            H5Dclose(ds_surface_);
            ds_surface_ = -1;
        }
        if (ds_water_ >= 0) {
            H5Dclose(ds_water_);
            ds_water_ = -1;
        }

        if (file_ >= 0) {
            H5Fclose(file_);
            file_ = -1;
        }

        closed_ = true;
    }

    void write_snapshot(const SavedStepMetadata& meta, const std::vector<float>& water_local_packed) {
        if (cfg_.io_mode == IoMode::ParallelHdf5) {
            append_parallel_3d_slice(ds_water_, water_local_packed);
            if (cfg_.write_surface) {
                std::vector<float> surface_local(water_local_packed.size(), 0.0F);
                for (std::size_t i = 0; i < water_local_packed.size(); ++i) {
                    surface_local[i] = terrain_local_packed_[i] + water_local_packed[i];
                }
                append_parallel_3d_slice(ds_surface_, surface_local);
            }
            append_parallel_step_metadata(meta);
            ++saved_count_;
            return;
        }

        std::vector<float> global_water = gather_global_field(water_local_packed, decomp_, comm_, decomp_.nz, decomp_.nx);
        if (decomp_.world_rank == 0) {
            append_rank0_3d_slice(ds_water_, global_water);
            if (cfg_.write_surface) {
                std::vector<float> global_surface(global_water.size(), 0.0F);
                for (std::size_t i = 0; i < global_water.size(); ++i) {
                    global_surface[i] = terrain_global_[i] + global_water[i];
                }
                append_rank0_3d_slice(ds_surface_, global_surface);
            }
            append_rank0_step_metadata(meta);
            ++saved_count_;
        }
        MPI_Bcast(&saved_count_, 1, MPI_UNSIGNED_LONG_LONG, 0, comm_);
    }

    std::size_t saved_count() const { return saved_count_; }

private:
    void setup_file() {
        int exists_int = 0;
        if (decomp_.world_rank == 0) {
            exists_int = std::filesystem::exists(cfg_.output_hdf5) ? 1 : 0;
        }
        MPI_Bcast(&exists_int, 1, MPI_INT, 0, comm_);
        if (exists_int == 1 && !cfg_.overwrite_output) {
            throw std::runtime_error(
                "Output file already exists. Use --overwrite-output to replace it: " + cfg_.output_hdf5);
        }

        if (cfg_.io_mode == IoMode::ParallelHdf5) {
            const hid_t fapl = create_parallel_fapl(comm_);
            file_ = H5Fcreate(cfg_.output_hdf5.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
            H5Pclose(fapl);
            require(file_ >= 0, "Failed to create output file (parallel): " + cfg_.output_hdf5);
        } else {
            if (decomp_.world_rank == 0) {
                file_ = H5Fcreate(cfg_.output_hdf5.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
                require(file_ >= 0, "Failed to create output file (rank0): " + cfg_.output_hdf5);
            }
        }

        write_metadata_attributes();
    }

    void write_metadata_attributes() {
        const bool write_on_this_rank = (cfg_.io_mode == IoMode::ParallelHdf5) || (decomp_.world_rank == 0);
        if (write_on_this_rank) {
            write_int_attribute("steps_total", cfg_.steps);
            write_int_attribute("save_every", cfg_.save_every);
            write_int_attribute("chunk_size", cfg_.chunk_size);
            write_int_attribute("triangular_scale", cfg_.triangular_scale);
            write_int_attribute("width", decomp_.nx);
            write_int_attribute("height", decomp_.nz);
            write_int_attribute("grid_width", decomp_.nx);
            write_int_attribute("grid_height", decomp_.nz);
            write_int_attribute("chunks_x", decomp_.chunks_x);
            write_int_attribute("chunks_z", decomp_.chunks_z);
            write_int_attribute("ranks_x", decomp_.ranks_x);
            write_int_attribute("ranks_z", decomp_.ranks_z);
            const int slurm_nodes = get_env_int("SLURM_NNODES", get_env_int("SLURM_JOB_NUM_NODES", -1));
            const int slurm_tasks = get_env_int("SLURM_NTASKS", -1);
            const int slurm_cores_per_task = get_env_int("SLURM_CPUS_PER_TASK", -1);
            write_int_attribute("nodes", slurm_nodes);
            write_int_attribute("tasks", slurm_tasks);
            write_int_attribute("cores_per_task", slurm_cores_per_task);
            write_int_attribute("slurm_nodes", slurm_nodes);
            write_int_attribute("slurm_tasks", slurm_tasks);
            write_int_attribute("slurm_cores_per_task", slurm_cores_per_task);
            write_string_attribute("save_mode", (cfg_.save_mode == SaveMode::Periodic) ? "periodic" : "triangular");
            write_string_attribute("boundary", "wraparound");
            write_string_attribute("update_rule", "quarter_diff_capped_and_scaled");
            write_string_attribute("io_mode", (cfg_.io_mode == IoMode::ParallelHdf5) ? "parallel_hdf5" : "rank0_gather");
            write_string_attribute("hdf5_xfer_mode",
                                   (cfg_.hdf5_xfer_mode == Hdf5XferMode::Collective) ? "collective" : "independent");
            write_string_attribute("sync_mode", sync_mode_to_string(cfg_.mpi_sync_mode));
            const std::string slurm_partition = get_env_string("SLURM_JOB_PARTITION", "");
            write_string_attribute("partition", slurm_partition);
            write_string_attribute("slurm_partition", slurm_partition);
        }
        MPI_Barrier(comm_);
    }

    hid_t create_write_dxpl() const {
        if (cfg_.hdf5_xfer_mode == Hdf5XferMode::Independent) {
            return create_independent_dxpl();
        }
        return create_collective_dxpl();
    }

    void write_int_attribute(const char* name, int value) {
        const hsize_t one = 1;
        const hid_t space = H5Screate_simple(1, &one, nullptr);
        require(space >= 0, "Failed to create attribute dataspace.");
        const hid_t attr = H5Acreate2(file_, name, H5T_NATIVE_INT, space, H5P_DEFAULT, H5P_DEFAULT);
        require(attr >= 0, std::string("Failed to create attribute: ") + name);
        check_h5(H5Awrite(attr, H5T_NATIVE_INT, &value), std::string("H5Awrite ") + name);
        H5Aclose(attr);
        H5Sclose(space);
    }

    void write_string_attribute(const char* name, const std::string& value) {
        const hsize_t one = 1;
        const hid_t space = H5Screate_simple(1, &one, nullptr);
        require(space >= 0, "Failed to create string attribute dataspace.");
        const hid_t type = H5Tcopy(H5T_C_S1);
        require(type >= 0, "Failed to create string attribute type.");
        check_h5(H5Tset_size(type, value.size()), "H5Tset_size string attribute");
        check_h5(H5Tset_strpad(type, H5T_STR_NULLTERM), "H5Tset_strpad string attribute");
        const hid_t attr = H5Acreate2(file_, name, type, space, H5P_DEFAULT, H5P_DEFAULT);
        require(attr >= 0, std::string("Failed to create string attribute: ") + name);
        check_h5(H5Awrite(attr, type, value.c_str()), std::string("H5Awrite string ") + name);
        H5Aclose(attr);
        H5Tclose(type);
        H5Sclose(space);
    }

    hid_t create_fixed_1d_dataset(const char* name, hid_t type) {
        const hsize_t dims[1] = {static_cast<hsize_t>(expected_snapshots_)};
        const hid_t space = H5Screate_simple(1, dims, nullptr);
        require(space >= 0, std::string("Failed to create dataspace for ") + name);
        const hid_t ds = H5Dcreate2(file_, name, type, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        require(ds >= 0, std::string("Failed to create dataset ") + name);
        H5Sclose(space);
        return ds;
    }

    void create_datasets_and_write_terrain() {
        if (cfg_.io_mode == IoMode::ParallelHdf5) {
            create_datasets_parallel();
            write_terrain_parallel();
            return;
        }

        if (decomp_.world_rank == 0) {
            create_datasets_rank0();
        }
        MPI_Barrier(comm_);
        write_terrain_rank0_gather();
    }

    void create_datasets_parallel() {
        const hsize_t terrain_dims[2] = {
            static_cast<hsize_t>(decomp_.nz),
            static_cast<hsize_t>(decomp_.nx),
        };
        const hid_t terrain_space = H5Screate_simple(2, terrain_dims, nullptr);
        require(terrain_space >= 0, "Failed to create terrain dataspace.");
        const hid_t terrain_ds = H5Dcreate2(file_, "terrain", H5T_IEEE_F32LE, terrain_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        require(terrain_ds >= 0, "Failed to create terrain dataset.");
        H5Dclose(terrain_ds);
        H5Sclose(terrain_space);

        const hsize_t water_dims[3] = {
            static_cast<hsize_t>(expected_snapshots_),
            static_cast<hsize_t>(decomp_.nz),
            static_cast<hsize_t>(decomp_.nx),
        };
        const hsize_t water_max[3] = {
            static_cast<hsize_t>(expected_snapshots_),
            static_cast<hsize_t>(decomp_.nz),
            static_cast<hsize_t>(decomp_.nx),
        };
        // Align chunking to simulation chunk grid to avoid cross-rank chunk contention.
        const hsize_t water_chunk[3] = {
            1,
            static_cast<hsize_t>(decomp_.chunk_size),
            static_cast<hsize_t>(decomp_.chunk_size),
        };

        const hid_t water_space = H5Screate_simple(3, water_dims, water_max);
        require(water_space >= 0, "Failed to create water dataspace.");
        const hid_t water_dcpl = H5Pcreate(H5P_DATASET_CREATE);
        require(water_dcpl >= 0, "Failed to create water DCPL.");
        check_h5(H5Pset_chunk(water_dcpl, 3, water_chunk), "H5Pset_chunk water");
        ds_water_ = H5Dcreate2(file_, "water", H5T_IEEE_F32LE, water_space, H5P_DEFAULT, water_dcpl, H5P_DEFAULT);
        require(ds_water_ >= 0, "Failed to create water dataset.");
        H5Pclose(water_dcpl);
        H5Sclose(water_space);

        if (cfg_.write_surface) {
            const hid_t surface_space = H5Screate_simple(3, water_dims, water_max);
            require(surface_space >= 0, "Failed to create surface dataspace.");
            const hid_t surface_dcpl = H5Pcreate(H5P_DATASET_CREATE);
            require(surface_dcpl >= 0, "Failed to create surface DCPL.");
            check_h5(H5Pset_chunk(surface_dcpl, 3, water_chunk), "H5Pset_chunk surface");
            ds_surface_ = H5Dcreate2(file_, "surface", H5T_IEEE_F32LE, surface_space, H5P_DEFAULT, surface_dcpl, H5P_DEFAULT);
            require(ds_surface_ >= 0, "Failed to create surface dataset.");
            H5Pclose(surface_dcpl);
            H5Sclose(surface_space);
        }

        ds_step_ = create_fixed_1d_dataset("step_index", H5T_STD_I32LE);
        ds_solver_type_ = create_fixed_1d_dataset("solver_type", H5T_STD_I32LE);
        ds_mass_ = create_fixed_1d_dataset("mass", H5T_IEEE_F64LE);
        ds_drift_ = create_fixed_1d_dataset("drift", H5T_IEEE_F64LE);
        ds_moved_this_step_ = create_fixed_1d_dataset("moved_this_step", H5T_IEEE_F64LE);
        ds_min_water_ = create_fixed_1d_dataset("min_water", H5T_IEEE_F32LE);
        ds_min_positive_water_ = create_fixed_1d_dataset("min_positive_water", H5T_IEEE_F32LE);
        ds_max_water_ = create_fixed_1d_dataset("max_water", H5T_IEEE_F32LE);
        ds_runtime_seconds_ = create_fixed_1d_dataset("runtime_seconds", H5T_IEEE_F64LE);
    }

    void write_terrain_parallel() {
        const hid_t terrain_ds = H5Dopen2(file_, "terrain", H5P_DEFAULT);
        require(terrain_ds >= 0, "Failed to open terrain dataset for write.");
        const hid_t file_space = H5Dget_space(terrain_ds);
        require(file_space >= 0, "Failed to get terrain file space.");

        const hsize_t start[2] = {
            static_cast<hsize_t>(decomp_.cell_z.begin),
            static_cast<hsize_t>(decomp_.cell_x.begin),
        };
        const hsize_t count[2] = {
            static_cast<hsize_t>(decomp_.local_nz),
            static_cast<hsize_t>(decomp_.local_nx),
        };

        check_h5(H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, nullptr, count, nullptr),
                 "H5Sselect_hyperslab terrain write");
        const hid_t mem_space = H5Screate_simple(2, count, nullptr);
        require(mem_space >= 0, "Failed to create terrain mem space.");
        const hid_t dxpl = create_collective_dxpl();
        check_h5(H5Dwrite(terrain_ds, H5T_NATIVE_FLOAT, mem_space, file_space, dxpl, terrain_local_packed_.data()),
                 "H5Dwrite terrain parallel");
        H5Pclose(dxpl);
        H5Sclose(mem_space);
        H5Sclose(file_space);
        H5Dclose(terrain_ds);
    }

    void create_datasets_rank0() {
        const hsize_t terrain_dims[2] = {
            static_cast<hsize_t>(decomp_.nz),
            static_cast<hsize_t>(decomp_.nx),
        };
        const hid_t terrain_space = H5Screate_simple(2, terrain_dims, nullptr);
        require(terrain_space >= 0, "Failed to create terrain dataspace (rank0). ");
        const hid_t terrain_ds = H5Dcreate2(file_, "terrain", H5T_IEEE_F32LE, terrain_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        require(terrain_ds >= 0, "Failed to create terrain dataset (rank0). ");
        H5Dclose(terrain_ds);
        H5Sclose(terrain_space);

        const hsize_t water_dims[3] = {
            static_cast<hsize_t>(expected_snapshots_),
            static_cast<hsize_t>(decomp_.nz),
            static_cast<hsize_t>(decomp_.nx),
        };
        const hsize_t water_max[3] = {
            static_cast<hsize_t>(expected_snapshots_),
            static_cast<hsize_t>(decomp_.nz),
            static_cast<hsize_t>(decomp_.nx),
        };
        const hsize_t water_chunk[3] = {1, std::min<hsize_t>(256, static_cast<hsize_t>(decomp_.nz)), std::min<hsize_t>(256, static_cast<hsize_t>(decomp_.nx))};

        const hid_t water_space = H5Screate_simple(3, water_dims, water_max);
        const hid_t water_dcpl = H5Pcreate(H5P_DATASET_CREATE);
        check_h5(H5Pset_chunk(water_dcpl, 3, water_chunk), "H5Pset_chunk water rank0");
        ds_water_ = H5Dcreate2(file_, "water", H5T_IEEE_F32LE, water_space, H5P_DEFAULT, water_dcpl, H5P_DEFAULT);
        require(ds_water_ >= 0, "Failed to create water dataset (rank0).");
        H5Pclose(water_dcpl);
        H5Sclose(water_space);

        if (cfg_.write_surface) {
            const hid_t surface_space = H5Screate_simple(3, water_dims, water_max);
            const hid_t surface_dcpl = H5Pcreate(H5P_DATASET_CREATE);
            check_h5(H5Pset_chunk(surface_dcpl, 3, water_chunk), "H5Pset_chunk surface rank0");
            ds_surface_ = H5Dcreate2(file_, "surface", H5T_IEEE_F32LE, surface_space, H5P_DEFAULT, surface_dcpl, H5P_DEFAULT);
            require(ds_surface_ >= 0, "Failed to create surface dataset (rank0).");
            H5Pclose(surface_dcpl);
            H5Sclose(surface_space);
        }

        ds_step_ = create_fixed_1d_dataset("step_index", H5T_STD_I32LE);
        ds_solver_type_ = create_fixed_1d_dataset("solver_type", H5T_STD_I32LE);
        ds_mass_ = create_fixed_1d_dataset("mass", H5T_IEEE_F64LE);
        ds_drift_ = create_fixed_1d_dataset("drift", H5T_IEEE_F64LE);
        ds_moved_this_step_ = create_fixed_1d_dataset("moved_this_step", H5T_IEEE_F64LE);
        ds_min_water_ = create_fixed_1d_dataset("min_water", H5T_IEEE_F32LE);
        ds_min_positive_water_ = create_fixed_1d_dataset("min_positive_water", H5T_IEEE_F32LE);
        ds_max_water_ = create_fixed_1d_dataset("max_water", H5T_IEEE_F32LE);
        ds_runtime_seconds_ = create_fixed_1d_dataset("runtime_seconds", H5T_IEEE_F64LE);
    }

    void write_terrain_rank0_gather() {
        std::vector<float> global_terrain = gather_global_field(terrain_local_packed_, decomp_, comm_, decomp_.nz, decomp_.nx);
        if (decomp_.world_rank == 0) {
            terrain_global_ = global_terrain;
            const hid_t terrain_ds = H5Dopen2(file_, "terrain", H5P_DEFAULT);
            require(terrain_ds >= 0, "Failed to open terrain dataset for rank0 write.");
            check_h5(H5Dwrite(terrain_ds, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, global_terrain.data()),
                     "H5Dwrite terrain rank0");
            H5Dclose(terrain_ds);
        }
        MPI_Barrier(comm_);
    }

    void append_parallel_3d_slice(hid_t ds, const std::vector<float>& local_packed) {
        require(saved_count_ < expected_snapshots_,
                "Snapshot index exceeded preallocated dataset size in parallel_hdf5 mode.");

        const hid_t file_space = H5Dget_space(ds);
        require(file_space >= 0, "Failed to get file space for parallel 3D append.");

        const hsize_t start[3] = {
            static_cast<hsize_t>(saved_count_),
            static_cast<hsize_t>(decomp_.cell_z.begin),
            static_cast<hsize_t>(decomp_.cell_x.begin),
        };
        const hsize_t count[3] = {
            1,
            static_cast<hsize_t>(decomp_.local_nz),
            static_cast<hsize_t>(decomp_.local_nx),
        };

        check_h5(H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, nullptr, count, nullptr),
                 "H5Sselect_hyperslab parallel 3D append");
        const hid_t mem_space = H5Screate_simple(3, count, nullptr);
        require(mem_space >= 0, "Failed to create mem space for parallel 3D append.");
        const hid_t dxpl = create_write_dxpl();
        check_h5(H5Dwrite(ds, H5T_NATIVE_FLOAT, mem_space, file_space, dxpl, local_packed.data()),
                 "H5Dwrite parallel 3D append");

        H5Pclose(dxpl);
        H5Sclose(mem_space);
        H5Sclose(file_space);
    }

    void append_parallel_scalar_root(hid_t ds, hid_t type, const void* value, const char* name) {
        require(saved_count_ < expected_snapshots_,
                std::string("Snapshot index exceeded preallocated size while writing ") + name);
        const hid_t file_space = H5Dget_space(ds);
        require(file_space >= 0, std::string("Failed to get file space for ") + name);

        hid_t mem_space = -1;
        const hid_t dxpl = create_write_dxpl();
        const hsize_t start[1] = {static_cast<hsize_t>(saved_count_)};
        const hsize_t count[1] = {1};

        if (decomp_.world_rank == 0) {
            check_h5(H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, nullptr, count, nullptr),
                     std::string("H5Sselect_hyperslab root ") + name);
            mem_space = H5Screate_simple(1, count, nullptr);
            require(mem_space >= 0, std::string("Failed to create memspace for root ") + name);
            check_h5(H5Dwrite(ds, type, mem_space, file_space, dxpl, value), std::string("H5Dwrite root ") + name);
        } else {
            check_h5(H5Sselect_none(file_space), std::string("H5Sselect_none non-root ") + name);
            mem_space = H5Screate(H5S_NULL);
            require(mem_space >= 0, std::string("Failed to create NULL memspace for non-root ") + name);
            check_h5(H5Dwrite(ds, type, mem_space, file_space, dxpl, nullptr),
                     std::string("H5Dwrite non-root ") + name);
        }

        H5Pclose(dxpl);
        H5Sclose(mem_space);
        H5Sclose(file_space);
    }

    void append_parallel_step_metadata(const SavedStepMetadata& meta) {
        append_parallel_scalar_root(ds_step_, H5T_NATIVE_INT, &meta.step, "step_index");
        append_parallel_scalar_root(ds_solver_type_, H5T_NATIVE_INT, &meta.solver_type, "solver_type");
        append_parallel_scalar_root(ds_mass_, H5T_NATIVE_DOUBLE, &meta.mass, "mass");
        append_parallel_scalar_root(ds_drift_, H5T_NATIVE_DOUBLE, &meta.drift, "drift");
        append_parallel_scalar_root(ds_moved_this_step_, H5T_NATIVE_DOUBLE, &meta.moved_this_step, "moved_this_step");
        append_parallel_scalar_root(ds_min_water_, H5T_NATIVE_FLOAT, &meta.min_water, "min_water");
        append_parallel_scalar_root(ds_min_positive_water_, H5T_NATIVE_FLOAT, &meta.min_positive_water, "min_positive_water");
        append_parallel_scalar_root(ds_max_water_, H5T_NATIVE_FLOAT, &meta.max_water, "max_water");
        append_parallel_scalar_root(ds_runtime_seconds_, H5T_NATIVE_DOUBLE, &meta.runtime_seconds, "runtime_seconds");
    }

    void append_rank0_3d_slice(hid_t ds, const std::vector<float>& global) {
        require(saved_count_ < expected_snapshots_,
                "Snapshot index exceeded preallocated dataset size in rank0 mode.");
        const hid_t file_space = H5Dget_space(ds);
        require(file_space >= 0, "Failed to get rank0 file space for 3D append.");
        const hsize_t start[3] = {static_cast<hsize_t>(saved_count_), 0, 0};
        const hsize_t count[3] = {1, static_cast<hsize_t>(decomp_.nz), static_cast<hsize_t>(decomp_.nx)};
        check_h5(H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, nullptr, count, nullptr),
                 "H5Sselect_hyperslab rank0 3D append");
        const hid_t mem_space = H5Screate_simple(3, count, nullptr);
        require(mem_space >= 0, "Failed to create rank0 memspace for 3D append.");
        check_h5(H5Dwrite(ds, H5T_NATIVE_FLOAT, mem_space, file_space, H5P_DEFAULT, global.data()),
                 "H5Dwrite rank0 3D append");
        H5Sclose(mem_space);
        H5Sclose(file_space);
    }

    void append_rank0_scalar(hid_t ds, hid_t type, const void* value, const char* name) {
        const hid_t file_space = H5Dget_space(ds);
        require(file_space >= 0, std::string("Failed to get rank0 file space for ") + name);
        const hsize_t start[1] = {static_cast<hsize_t>(saved_count_)};
        const hsize_t count[1] = {1};
        check_h5(H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, nullptr, count, nullptr),
                 std::string("H5Sselect_hyperslab rank0 ") + name);
        const hid_t mem_space = H5Screate_simple(1, count, nullptr);
        require(mem_space >= 0, std::string("Failed to create rank0 memspace for ") + name);
        check_h5(H5Dwrite(ds, type, mem_space, file_space, H5P_DEFAULT, value),
                 std::string("H5Dwrite rank0 ") + name);
        H5Sclose(mem_space);
        H5Sclose(file_space);
    }

    void append_rank0_step_metadata(const SavedStepMetadata& meta) {
        append_rank0_scalar(ds_step_, H5T_NATIVE_INT, &meta.step, "step_index");
        append_rank0_scalar(ds_solver_type_, H5T_NATIVE_INT, &meta.solver_type, "solver_type");
        append_rank0_scalar(ds_mass_, H5T_NATIVE_DOUBLE, &meta.mass, "mass");
        append_rank0_scalar(ds_drift_, H5T_NATIVE_DOUBLE, &meta.drift, "drift");
        append_rank0_scalar(ds_moved_this_step_, H5T_NATIVE_DOUBLE, &meta.moved_this_step, "moved_this_step");
        append_rank0_scalar(ds_min_water_, H5T_NATIVE_FLOAT, &meta.min_water, "min_water");
        append_rank0_scalar(ds_min_positive_water_, H5T_NATIVE_FLOAT, &meta.min_positive_water, "min_positive_water");
        append_rank0_scalar(ds_max_water_, H5T_NATIVE_FLOAT, &meta.max_water, "max_water");
        append_rank0_scalar(ds_runtime_seconds_, H5T_NATIVE_DOUBLE, &meta.runtime_seconds, "runtime_seconds");
    }

private:
    Config cfg_;
    Decomposition decomp_;
    MPI_Comm comm_;

    std::vector<float> terrain_local_packed_;
    std::vector<float> terrain_global_;

    hid_t file_ = -1;
    hid_t ds_water_ = -1;
    hid_t ds_surface_ = -1;
    hid_t ds_step_ = -1;
    hid_t ds_solver_type_ = -1;
    hid_t ds_mass_ = -1;
    hid_t ds_drift_ = -1;
    hid_t ds_moved_this_step_ = -1;
    hid_t ds_min_water_ = -1;
    hid_t ds_min_positive_water_ = -1;
    hid_t ds_max_water_ = -1;
    hid_t ds_runtime_seconds_ = -1;

    std::size_t saved_count_ = 0;
    std::size_t expected_snapshots_ = 0;
    bool closed_ = false;
};

static Decomposition build_decomposition(
    const Config& cfg,
    MPI_Comm world,
    int world_rank,
    int world_size,
    int nx,
    int nz) {
    Decomposition d;
    d.world_rank = world_rank;
    d.world_size = world_size;
    d.nx = nx;
    d.nz = nz;
    d.chunk_size = cfg.chunk_size;

    require(nx % cfg.chunk_size == 0, "Grid width must be divisible by chunk size. Got nx=" + std::to_string(nx) + " and chunk_size=" + std::to_string(cfg.chunk_size) + " " + std::to_string(nx / cfg.chunk_size) + " chunks.");
    require(nz % cfg.chunk_size == 0, "Grid height must be divisible by chunk size. Got nz=" + std::to_string(nz) + " and chunk_size=" + std::to_string(cfg.chunk_size) + " " + std::to_string(nz / cfg.chunk_size) + " chunks.");
    d.chunks_x = nx / cfg.chunk_size;
    d.chunks_z = nz / cfg.chunk_size;

    int dims[2] = {0, 0}; // [z, x]
    if (cfg.rank_grid_x == 0 && cfg.rank_grid_z == 0) {
        dims[0] = 0;
        dims[1] = 0;
        MPI_Dims_create(world_size, 2, dims);
        // Check if the factorization matches chunk grid; swap if needed.
        if (dims[0] > d.chunks_z && dims[1] <= d.chunks_z && dims[0] <= d.chunks_x) {
            std::swap(dims[0], dims[1]);
        }
    } else if (cfg.rank_grid_x > 0 && cfg.rank_grid_z > 0) {
        require(cfg.rank_grid_x * cfg.rank_grid_z == world_size,
                "rank_grid_x * rank_grid_z must equal MPI world size.");
        dims[0] = cfg.rank_grid_z;
        dims[1] = cfg.rank_grid_x;
    } else if (cfg.rank_grid_x > 0) {
        require(world_size % cfg.rank_grid_x == 0,
                "world_size must be divisible by rank_grid_x when rank_grid_z is omitted.");
        dims[1] = cfg.rank_grid_x;
        dims[0] = world_size / cfg.rank_grid_x;
    } else {
        require(world_size % cfg.rank_grid_z == 0,
                "world_size must be divisible by rank_grid_z when rank_grid_x is omitted.");
        dims[0] = cfg.rank_grid_z;
        dims[1] = world_size / cfg.rank_grid_z;
    }

    require(dims[0] <= d.chunks_z, "Rank grid Z is larger than chunk grid Z; would create empty ranks. Got dims[0]=" + std::to_string(dims[0]) + " and d.chunks_z=" + std::to_string(d.chunks_z) + ". In total we have ranks: " + std::to_string(dims[0]) + "x" + std::to_string(dims[1]) + " which doesnt fit " + std::to_string(d.chunks_z) + "x" + std::to_string(d.chunks_x));
    require(dims[1] <= d.chunks_x, "Rank grid X is larger than chunk grid X; would create empty ranks. Got dims[1]=" + std::to_string(dims[1]) + " and d.chunks_x=" + std::to_string(d.chunks_x) + ". In total we have ranks: " + std::to_string(dims[0]) + "x" + std::to_string(dims[1]) + " which doesnt fit " + std::to_string(d.chunks_z) + "x" + std::to_string(d.chunks_x));

    int periods[2] = {1, 1};
    MPI_Comm cart = MPI_COMM_NULL;
    MPI_Cart_create(world, 2, dims, periods, 0, &cart);
    require(cart != MPI_COMM_NULL, "MPI_Cart_create failed.");
    d.cart_comm = cart;

    MPI_Comm_rank(cart, &d.cart_rank);
    int coords[2] = {0, 0};
    MPI_Cart_coords(cart, d.cart_rank, 2, coords);

    d.coords_z = coords[0];
    d.coords_x = coords[1];
    d.ranks_z = dims[0];
    d.ranks_x = dims[1];

    MPI_Cart_shift(cart, 0, 1, &d.north, &d.south);
    MPI_Cart_shift(cart, 1, 1, &d.west, &d.east);

    d.chunk_z = partition_range(d.chunks_z, d.ranks_z, d.coords_z);
    d.chunk_x = partition_range(d.chunks_x, d.ranks_x, d.coords_x);

    d.cell_z.begin = d.chunk_z.begin * cfg.chunk_size;
    d.cell_z.end = d.chunk_z.end * cfg.chunk_size;
    d.cell_x.begin = d.chunk_x.begin * cfg.chunk_size;
    d.cell_x.end = d.chunk_x.end * cfg.chunk_size;

    d.local_nz = d.cell_z.size();
    d.local_nx = d.cell_x.size();

    require(d.local_nz > 0 && d.local_nx > 0, "Local decomposition produced an empty rank.");
    return d;
}

struct GlobalStats {
    double mass = 0.0;
    double moved = 0.0;
    float min_water = 0.0F;
    float max_water = 0.0F;
    float min_positive_water = 0.0F;
};

static GlobalStats reduce_global_stats(
    const std::vector<float>& water,
    const Decomposition& decomp,
    double moved_this_step) {
    const int pitch = decomp.local_nx + 2;

    double local_mass = 0.0;
    float local_min = std::numeric_limits<float>::infinity();
    float local_max = -std::numeric_limits<float>::infinity();
    float local_min_pos = std::numeric_limits<float>::infinity();

    for (int i = 1; i <= decomp.local_nz; ++i) {
        for (int j = 1; j <= decomp.local_nx; ++j) {
            const float v = water[local_index(i, j, pitch)];
            local_mass += static_cast<double>(v);
            if (v < local_min) {
                local_min = v;
            }
            if (v > local_max) {
                local_max = v;
            }
            if (v > 0.0F && v < local_min_pos) {
                local_min_pos = v;
            }
        }
    }

    GlobalStats s;
    MPI_Allreduce(&local_mass, &s.mass, 1, MPI_DOUBLE, MPI_SUM, decomp.cart_comm);
    MPI_Allreduce(&moved_this_step, &s.moved, 1, MPI_DOUBLE, MPI_SUM, decomp.cart_comm);
    MPI_Allreduce(&local_min, &s.min_water, 1, MPI_FLOAT, MPI_MIN, decomp.cart_comm);
    MPI_Allreduce(&local_max, &s.max_water, 1, MPI_FLOAT, MPI_MAX, decomp.cart_comm);
    MPI_Allreduce(&local_min_pos, &s.min_positive_water, 1, MPI_FLOAT, MPI_MIN, decomp.cart_comm);

    if (!std::isfinite(s.min_positive_water)) {
        s.min_positive_water = 0.0F;
    }
    return s;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    bool USE_SMARTSIM;

    try {
        const Config cfg = parse_args(argc, argv, world_rank, world_size);

        if (cfg.print_build_timestamp) {
            if (world_rank == 0) {
                // Compile-time stamp: constructed from __DATE__ and __TIME__
                std::string build_date(__DATE__);
                std::string build_time(__TIME__);
                std::cout << "Build timestamp: " << build_date << " " << build_time << std::endl;
            }
            MPI_Finalize();
            return 0;
        }

        if (cfg.io_mode == IoMode::ParallelHdf5) {
#ifndef H5_HAVE_PARALLEL
            throw std::runtime_error("--io-mode parallel_hdf5 requested but HDF5 is not built with parallel support.");
#endif
        }

        // Prepare smartsim
        if (getenv("SSDB") == nullptr) {
            //setenv("SSDB", "127.0.0.1:6379", 0);
            std::cout << "SSDB not set. Disabling Smartsim" << std::endl;
            USE_SMARTSIM = false;
        } else {
            std::cout << "Expecting smartsim database/orchestrator at " << getenv("SSDB") << std::endl;
            USE_SMARTSIM = true;
        }

        SmartRedis::Client* client = nullptr;
        
        if (USE_SMARTSIM) {
            client = new SmartRedis::Client("terrain_solver_" + std::to_string(world_rank));

            std::cout << "Successfully connected to SmartRedis database as terrain_solver_" << std::to_string(world_rank) << std::endl;

            if (world_rank == 0) {
                if (!std::filesystem::exists(cfg.model_path)) {
                    throw std::runtime_error("SmartRedis model file not found: " + cfg.model_path + " (cwd=" + std::filesystem::current_path().string() + ")");
                }
                //set_model_from_file(const std::string &name, const std::string &model_file, const std::string &backend, const std::string &device, int batch_size = 0, int min_batch_size = 0, int min_batch_timeout = 0, const std::string &tag = "", const std::vector<std::string> &inputs = std::vector<std::string>(), const std::vector<std::string> &outputs = std::vector<std::string>())
                if (cfg.device == "GPU") {
                    if (cfg.gpus_per_node == 0) {
                        throw std::runtime_error("GPU device specified but --gpus-per-node is 0.");
                    }
                    std::cout << "Loading model onto GPU with multi-GPU support, gpus_per_node=" << cfg.gpus_per_node << std::endl;
                    client->set_model_from_file_multigpu("water_step_model_gpu", cfg.model_path, "TORCH", 0, cfg.gpus_per_node);
                } else {
                    client->set_model_from_file("water_step_model", cfg.model_path, "TORCH", cfg.device);
                }
            }


        }

        std::size_t nz_u = 0;
        std::size_t nx_u = 0;
        if (cfg.io_mode == IoMode::Rank0Gather) {
            read_global_dims_serial(cfg.input_hdf5, nz_u, nx_u, world_rank, MPI_COMM_WORLD);
        } else {
            read_global_dims_parallel(cfg.input_hdf5, nz_u, nx_u, MPI_COMM_WORLD);
        }

        const int nz = static_cast<int>(nz_u);
        const int nx = static_cast<int>(nx_u);

        Decomposition decomp = build_decomposition(cfg, MPI_COMM_WORLD, world_rank, world_size, nx, nz);

        if (world_rank == 0) {
            std::cout << "Loaded grid metadata: nx=" << nx
                      << ", nz=" << nz
                      << ", chunk_size=" << cfg.chunk_size
                      << ", chunks_x=" << decomp.chunks_x
                      << ", chunks_z=" << decomp.chunks_z
                      << ", ranks_x=" << decomp.ranks_x
                      << ", ranks_z=" << decomp.ranks_z
                      << ", world_size=" << world_size
                      << ", io_mode=" << (cfg.io_mode == IoMode::ParallelHdf5 ? "parallel_hdf5" : "rank0_gather")
                      << ", save_mode=" << (cfg.save_mode == SaveMode::Periodic ? "periodic" : "triangular")
                      << ", triangular_scale=" << cfg.triangular_scale
                      << ", hdf5_xfer_mode=" << (cfg.hdf5_xfer_mode == Hdf5XferMode::Collective ? "collective" : "independent")
                      << ", sync_mode="
                      << (cfg.mpi_sync_mode == SyncMode::None ? "none" : (cfg.mpi_sync_mode == SyncMode::Step ? "step" : "report"))
                      << std::endl;
        }

        std::vector<float> terrain_local_packed;
        std::vector<float> water_local_packed;
        if (cfg.io_mode == IoMode::Rank0Gather) {
            terrain_local_packed = read_local_2d_rank0_gather(cfg.input_hdf5, "terrain", decomp, nz_u, nx_u);
            water_local_packed   = read_local_2d_rank0_gather(cfg.input_hdf5, "water_init", decomp, nz_u, nx_u);
        } else {
            terrain_local_packed = read_local_2d_parallel(cfg.input_hdf5, "terrain", decomp, nz_u, nx_u);
            water_local_packed   = read_local_2d_parallel(cfg.input_hdf5, "water_init", decomp, nz_u, nx_u);
        }

        const int pitch = decomp.local_nx + 2;
        std::vector<float> terrain(static_cast<std::size_t>(decomp.local_nz + 2) * static_cast<std::size_t>(decomp.local_nx + 2), 0.0F);
        std::vector<float> water(static_cast<std::size_t>(decomp.local_nz + 2) * static_cast<std::size_t>(decomp.local_nx + 2), 0.0F);
        std::vector<float> next = water;

        copy_packed_to_with_halo(terrain_local_packed, terrain, decomp.local_nz, decomp.local_nx);
        copy_packed_to_with_halo(water_local_packed, water, decomp.local_nz, decomp.local_nx);
        exchange_halo_1cell(terrain, decomp, 1000);

        // Split the terrain data into width x height many 3x3 tiles (one tile per cell), and pack each tile into a contiguous 9-float array. This is the format expected by the ML model. 

        if (USE_SMARTSIM) {

            std::cout << "Preparing terrain tiles for ML model" << std::endl;

            const int BATCH_SIZE = decomp.local_nz * decomp.local_nx;

            std::vector<float***> tiles_batch(static_cast<std::size_t>(BATCH_SIZE));
            std::vector<float**> tile_channels(static_cast<std::size_t>(BATCH_SIZE));
            std::vector<float*> tile_rows(static_cast<std::size_t>(BATCH_SIZE) * 3);
            std::vector<float> terrain_tiles(static_cast<std::size_t>(BATCH_SIZE) * 9, 0.0F);

            for (int i = 0; i < decomp.local_nz; ++i) {
                for (int j = 0; j < decomp.local_nx; ++j) {
                    const int ii = i + 1;
                    const int jj = j + 1;
                    const std::size_t b = static_cast<std::size_t>(i * decomp.local_nx + j);
                    float* tile_flat = &terrain_tiles[b * 9];
                    tiles_batch[b] = &tile_channels[b];
                    tile_channels[b] = &tile_rows[b * 3];
                    for (int di = -1; di <= 1; ++di) {
                        tile_rows[b * 3 + static_cast<std::size_t>(di + 1)] = tile_flat + static_cast<std::size_t>((di + 1) * 3);
                        for (int dj = -1; dj <= 1; ++dj) {
                            const int n_i = ii + di;
                            const int n_j = jj + dj;
                            const std::size_t tile_idx = static_cast<std::size_t>((di + 1) * 3 + (dj + 1));
                            //tile_flat[tile_idx] = terrain[local_index(n_i, n_j, pitch)];
                            //tiles[di + 1][dj + 1] = terrain[local_index(n_i, n_j, pitch)];
                            tile_flat[tile_idx] = terrain[local_index(n_i, n_j, pitch)];
                        }
                    }
                }
            }


            client->put_tensor(
                "terrain_tile_" + std::to_string(decomp.world_rank),
                const_cast<float****>(tiles_batch.data()),
                std::vector<size_t>{static_cast<size_t>(BATCH_SIZE), 1, 3, 3},
                SRTensorType::SRTensorTypeFloat,
                SRMemoryLayout::SRMemLayoutNested);

            std::cout << "Finished preparing terrain tiles for ML model" << std::endl;

        }

        StepScratch step_scratch(decomp.local_nz, decomp.local_nx);

        MlNestedBatchView ml_batch_view_water;
        MlNestedBatchView ml_batch_view_next;
        std::vector<float> ml_tile_output;
        if (USE_SMARTSIM) {
            ml_batch_view_water = create_ml_nested_batch_view(water, decomp);
            ml_batch_view_next = create_ml_nested_batch_view(next, decomp);
            ml_tile_output.resize(static_cast<std::size_t>(decomp.local_nz) * static_cast<std::size_t>(decomp.local_nx), 0.0F);
        }

        const GlobalStats initial_stats = reduce_global_stats(water, decomp, 0.0);
        if (world_rank == 0) {
            std::cout << "Initial water mass: " << initial_stats.mass << std::endl;
        }

        std::size_t saved_count = 0;
        {
            // Ensure all HDF5 handles are closed before MPI_Finalize().
            TrajectoryWriter writer(cfg, decomp, terrain_local_packed, decomp.cart_comm);
            SavedStepMetadata initial_meta;
            initial_meta.step = 0;
            initial_meta.solver_type = 0;
            initial_meta.mass = initial_stats.mass;
            initial_meta.drift = 0.0;
            initial_meta.moved_this_step = 0.0;
            initial_meta.min_water = initial_stats.min_water;
            initial_meta.min_positive_water = initial_stats.min_positive_water;
            initial_meta.max_water = initial_stats.max_water;
            initial_meta.runtime_seconds = 0.0;
            writer.write_snapshot(initial_meta, pack_interior(water, decomp.local_nz, decomp.local_nx));

            std::vector<int> milestone_steps;
            milestone_steps.reserve(10);
            for (int i = 1; i <= 10; ++i) {
                const int milestone = static_cast<int>(
                    std::ceil((static_cast<double>(i) * static_cast<double>(cfg.steps)) / 10.0));
                milestone_steps.push_back(std::max(1, milestone));
            }
            std::sort(milestone_steps.begin(), milestone_steps.end());
            milestone_steps.erase(std::unique(milestone_steps.begin(), milestone_steps.end()), milestone_steps.end());

            std::size_t next_milestone_idx = 0;
            const auto sim_start = std::chrono::steady_clock::now();
            auto last_report_time = sim_start;
            const auto report_max_silence = std::chrono::minutes(5);
            int next_triangular_save_step = cfg.triangular_scale;
            int triangular_k = 2;

            for (int step = 1; step <= cfg.steps; ++step) {
                exchange_halo_1cell(water, decomp, 2000);

                double moved_this_step_local = 0.0;

                auto start_this_step = std::chrono::steady_clock::now();
                const bool use_ml_step = ((step % 2) == 0) && USE_SMARTSIM;
                
                if (use_ml_step) {

                    const MlNestedBatchView& active_ml_view =
                        (water.data() == ml_batch_view_water.base_data) ? ml_batch_view_water : ml_batch_view_next;

                    moved_this_step_local = compute_local_step_ml(
                        client,
                        terrain,
                        water,
                        next,
                        decomp,
                        cfg.clamp_epsilon,
                        cfg,
                        active_ml_view,
                        ml_tile_output);

                } else {


                    moved_this_step_local = compute_local_step(
                        terrain,
                        water,
                        next,
                        decomp,
                        cfg.clamp_epsilon,
                        step_scratch);

                }

                if (world_rank == 0) {
                    const auto end_this_step = std::chrono::steady_clock::now();
                    const auto step_ms = std::chrono::duration_cast<std::chrono::nanoseconds>(end_this_step - start_this_step).count();
                    std::cout << "Step " << step << ", " << (use_ml_step ? "ML" : "Regular")
                              << ", local moved: " << moved_this_step_local
                              << ", time: " << step_ms / 1e6 << " ms" << std::endl;
                    std::cout << "STEP_TIMING step=" << step
                              << " solver=" << (use_ml_step ? "ML" : "Regular")
                              << " step_ms=" << step_ms / 1e6
                              << " local_moved=" << moved_this_step_local
                              << std::endl;
                }

                bool do_save = false;
                if (cfg.save_mode == SaveMode::Periodic) {
                    do_save = ((step % cfg.save_every) == 0);
                } else {
                    if (step == next_triangular_save_step) {
                        do_save = true;
                        next_triangular_save_step += triangular_k * cfg.triangular_scale;
                        ++triangular_k;
                    }
                }
                if (step == cfg.steps) {
                    do_save = true; // always save final state
                }

                bool do_report = false;
                std::chrono::steady_clock::time_point now;
                if (world_rank == 0) {
                    now = std::chrono::steady_clock::now();
                    bool reached_milestone = false;
                    while (next_milestone_idx < milestone_steps.size() && step >= milestone_steps[next_milestone_idx]) {
                        reached_milestone = true;
                        ++next_milestone_idx;
                    }
                    const bool reached_heartbeat = (now - last_report_time) >= report_max_silence;
                    do_report = reached_milestone || reached_heartbeat || (step == cfg.steps);
                }
                int do_report_int = do_report ? 1 : 0;
                MPI_Bcast(&do_report_int, 1, MPI_INT, 0, decomp.cart_comm);
                do_report = (do_report_int != 0);

                if (cfg.mpi_sync_mode == SyncMode::Report && (do_report || do_save)) {
                    MPI_Barrier(decomp.cart_comm);
                }

                water.swap(next);

                const bool need_stats = do_report || do_save;
                GlobalStats stats{};
                if (need_stats) {
                    stats = reduce_global_stats(water, decomp, moved_this_step_local);
                }

                if (do_save) {
                    if (world_rank == 0 && step == cfg.steps) {
                        std::cout << "[shutdown] starting final snapshot write" << std::endl;
                    }
                    double runtime_seconds = 0.0;
                    if (world_rank == 0) {
                        runtime_seconds = std::chrono::duration<double>(now - sim_start).count();
                    }
                    MPI_Bcast(&runtime_seconds, 1, MPI_DOUBLE, 0, decomp.cart_comm);
                    SavedStepMetadata meta;
                    meta.step = step;
                    meta.solver_type = use_ml_step ? 2 : 1;
                    meta.mass = stats.mass;
                    meta.drift = stats.mass - initial_stats.mass;
                    meta.moved_this_step = stats.moved;
                    meta.min_water = stats.min_water;
                    meta.min_positive_water = stats.min_positive_water;
                    meta.max_water = stats.max_water;
                    meta.runtime_seconds = runtime_seconds;
                    writer.write_snapshot(meta, pack_interior(water, decomp.local_nz, decomp.local_nx));
                    if (world_rank == 0 && step == cfg.steps) {
                        std::cout << "[shutdown] finished final snapshot write" << std::endl;

                        std::cout << "Total runtime: " << format_duration(std::chrono::duration_cast<std::chrono::seconds>(now - sim_start)) << std::endl;
                        std::cout << "Total prepare data time: " << format_duration(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::microseconds(total_prepare_data_time))) << std::endl;
                        std::cout << "Total put tensor time: " << format_duration(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::microseconds(total_put_tensor_time))) << std::endl;
                        std::cout << "Total run model time: " << format_duration(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::microseconds(total_run_model_time))) << std::endl;
                        std::cout << "Total unpack time: " << format_duration(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::microseconds(total_unpack_time))) << std::endl;
                        std::cout << "Total cleanup time: " << format_duration(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::microseconds(total_cleanup_time))) << std::endl;
                        std::cout << "Total ML step wall time: " << format_duration(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::microseconds(total_ml_step_wall_time))) << std::endl;
                    }
                }

                if (do_report) {
                    if (world_rank == 0) {
                        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - sim_start);
                        std::chrono::seconds eta(0);
                        if (step > 0 && step < cfg.steps) {
                            const double elapsed_s = static_cast<double>(elapsed.count());
                            const double avg_s_per_step = elapsed_s / static_cast<double>(step);
                            const double remaining_s = avg_s_per_step * static_cast<double>(cfg.steps - step);
                            eta = std::chrono::seconds(static_cast<long long>(std::llround(remaining_s)));
                        }

                        std::cout << "Step " << step << "/" << cfg.steps
                                  << " | mass=" << stats.mass
                                  << " | drift=" << (stats.mass - initial_stats.mass)
                                  << " | moved_this_step=" << stats.moved
                                  << " | min_water=" << stats.min_water
                                  << " | min_positive_water=" << stats.min_positive_water
                                  << " | max_water=" << stats.max_water
                                  << " | runtime=" << format_duration(elapsed)
                                  << " | eta=" << format_duration(eta)
                                  << std::endl;
                        last_report_time = now;
                    }
                }

                if (cfg.mpi_sync_mode == SyncMode::Step) {
                    MPI_Barrier(decomp.cart_comm);
                }
            }

            if (world_rank == 0) {
                std::cout << "[shutdown] starting trajectory writer close" << std::endl;
            }
            writer.close();
            if (world_rank == 0) {
                std::cout << "[shutdown] finished trajectory writer close" << std::endl;
            }
            saved_count = writer.saved_count();
        }

        if (world_rank == 0) {
            std::cout << "Finished. Saved snapshots: " << saved_count
                      << " to " << cfg.output_hdf5 << std::endl;
            std::cout << "[shutdown] entering MPI_Comm_free / MPI_Finalize" << std::endl;
        }

        if (decomp.cart_comm != MPI_COMM_NULL) {
            MPI_Comm_free(&decomp.cart_comm);
        }
        MPI_Finalize();
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "ERROR: " << ex.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
