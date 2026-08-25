#pragma once

#include "application/ml_coupling_application.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef USE_SCOREP
#include <scorep/SCOREP_User.h>
#endif

// @registry_name: MLCouplingApplicationTerrainSolver
// @registry_aliases: terrain-solver-app, terrain_solver_app
// @registry_description: Terrain solver coupling application performing stencil extraction and field update.
template <typename CouplingInput,
          typename CouplingOutput,
          typename LibraryInput = CouplingInput,
          typename LibraryOutput = CouplingOutput>
class MLCouplingApplicationTerrainSolver : public MLCouplingApplication<CouplingInput,
                                                                        CouplingOutput,
                                                                        LibraryInput,
                                                                        LibraryOutput>
{
public:
    MLCouplingApplicationTerrainSolver(MLCouplingData<CouplingInput> coupling_input,
                                       MLCouplingData<CouplingOutput> coupling_output,
                                       std::string model_io_layout = "split_3x3")
        : MLCouplingApplication<CouplingInput, CouplingOutput, LibraryInput, LibraryOutput>(
              std::move(coupling_input), std::move(coupling_output), nullptr),
          model_io_layout_(std::move(model_io_layout))
    {
        init_internal_buffers();
    }

    MLCouplingApplicationTerrainSolver(MLCouplingData<CouplingInput> coupling_input,
                                       MLCouplingData<LibraryInput> library_input,
                                       MLCouplingData<LibraryOutput> library_output,
                                       MLCouplingData<CouplingOutput> coupling_output,
                                       std::string model_io_layout = "split_3x3")
        : MLCouplingApplication<CouplingInput, CouplingOutput, LibraryInput, LibraryOutput>(
              std::move(coupling_input),
              std::move(library_input),
              std::move(library_output),
              std::move(coupling_output),
              nullptr),
          model_io_layout_(std::move(model_io_layout))
    {
        init_internal_buffers();
    }

    double get_last_moved() const { return last_moved_; }
    int get_active_buffer_index() const { return active_buffer_idx_; }
    void set_active_buffer_index(int idx) { active_buffer_idx_ = idx; }

protected:
    MLCouplingData<LibraryInput>
    preprocess_coupling_input(MLCouplingData<CouplingInput> input) override
    {
        if (input.size() < 2)
        {
            throw std::runtime_error("MLCouplingApplicationTerrainSolver requires at least 2 input tensors.");
        }

        const auto& dims = input[0].dimensions();
        if (dims.size() >= 2)
        {
            const int nz = dims[0] - 2;
            const int nx = dims[1] - 2;
            const int pitch = dims[1];
            const std::size_t bs = static_cast<std::size_t>(std::max(0, nz)) * static_cast<std::size_t>(std::max(0, nx));

            if (nz != local_nz_ || nx != local_nx_ || bs != batch_size_)
            {
                local_nz_ = nz;
                local_nx_ = nx;
                pitch_ = pitch;
                batch_size_ = bs;
                this->coupling_input = input;
                init_internal_buffers();
            }
        }

        const bool is_ping_pong = (input.size() >= 3);
        const int water_idx = is_ping_pong ? active_buffer_idx_ : 0;
        const int terrain_idx = is_ping_pong ? 2 : 1;

        const CouplingInput* water_ptr = static_cast<const CouplingInput*>(input[water_idx].root());
        const CouplingInput* terrain_ptr = static_cast<const CouplingInput*>(input[terrain_idx].root());

        const bool use_flat = (model_io_layout_ == "flat_contiguous");

        if (use_flat)
        {
            std::fill(flat_input_buffer_.begin(), flat_input_buffer_.end(), static_cast<LibraryInput>(0));
            for (std::size_t k = 0; k < batch_size_; ++k)
            {
                const int i = static_cast<int>(k / static_cast<std::size_t>(local_nx_));
                const int j = static_cast<int>(k % static_cast<std::size_t>(local_nx_));
                const int ii = i + 1;
                const int jj = j + 1;
                LibraryInput* packed = flat_input_buffer_.data() + (k * 18);

                std::size_t idx = 0;
                for (int di = -1; di <= 1; ++di)
                {
                    for (int dj = -1; dj <= 1; ++dj)
                    {
                        const int n_i = ii + di;
                        const int n_j = jj + dj;
                        packed[idx++] = static_cast<LibraryInput>(water_ptr[n_i * pitch_ + n_j]);
                    }
                }
                for (int di = -1; di <= 1; ++di)
                {
                    for (int dj = -1; dj <= 1; ++dj)
                    {
                        const int n_i = ii + di;
                        const int n_j = jj + dj;
                        packed[idx++] = static_cast<LibraryInput>(terrain_ptr[n_i * pitch_ + n_j]);
                    }
                }
            }
        }
        else
        {
            for (std::size_t k = 0; k < batch_size_; ++k)
            {
                const int i = static_cast<int>(k / static_cast<std::size_t>(local_nx_));
                const int j = static_cast<int>(k % static_cast<std::size_t>(local_nx_));
                const int ii = i + 1;
                const int jj = j + 1;

                for (int di = -1; di <= 1; ++di)
                {
                    const int n_i = ii + di;
                    const int n_j_start = jj - 1;
                    water_rows_[k * 3 + static_cast<std::size_t>(di + 1)] =
                        const_cast<LibraryInput*>(reinterpret_cast<const LibraryInput*>(water_ptr + (n_i * pitch_ + n_j_start)));
                    terrain_rows_[k * 3 + static_cast<std::size_t>(di + 1)] =
                        const_cast<LibraryInput*>(reinterpret_cast<const LibraryInput*>(terrain_ptr + (n_i * pitch_ + n_j_start)));
                }
            }
        }

        std::fill(model_output_buffer_.begin(), model_output_buffer_.end(), static_cast<LibraryOutput>(0));
        return this->library_input;
    }

    MLCouplingData<CouplingOutput>
    postprocess_library_output(MLCouplingData<LibraryOutput> output_data_before_postprocessing) override
    {
        if (this->coupling_input.size() < 1 || this->coupling_output.size() < 1)
        {
            throw std::runtime_error("MLCouplingApplicationTerrainSolver: coupling_input or coupling_output buffer missing.");
        }

        const bool is_ping_pong = (this->coupling_input.size() >= 3 && this->coupling_output.size() >= 3);
        const int water_in_idx = is_ping_pong ? active_buffer_idx_ : 0;
        const int water_out_idx = is_ping_pong ? (1 - active_buffer_idx_) : 0;
        const int moved_out_idx = is_ping_pong ? 2 : 1;

        const CouplingInput* water_ptr = static_cast<const CouplingInput*>(this->coupling_input[water_in_idx].root());
        CouplingOutput* next_ptr = static_cast<CouplingOutput*>(this->coupling_output[water_out_idx].root());
        const LibraryOutput* model_out_ptr = static_cast<const LibraryOutput*>(output_data_before_postprocessing[0].root());

        const std::size_t total_grid_cells = static_cast<std::size_t>(local_nz_ + 2) * static_cast<std::size_t>(pitch_);
        std::copy(water_ptr, water_ptr + total_grid_cells, next_ptr);

        double moved = 0.0;
        for (int i = 0; i < local_nz_; ++i)
        {
            for (int j = 0; j < local_nx_; ++j)
            {
                const std::size_t k = static_cast<std::size_t>(i * local_nx_ + j);
                const std::size_t grid_idx = static_cast<std::size_t>((i + 1) * pitch_ + (j + 1));
                const auto pred = static_cast<CouplingOutput>(model_out_ptr[k]);
                next_ptr[grid_idx] = pred;
                moved += std::max(static_cast<double>(pred) - static_cast<double>(water_ptr[grid_idx]), 0.0);
            }
        }

        last_moved_ = moved;

        if (this->coupling_output.size() > static_cast<std::size_t>(moved_out_idx))
        {
            CouplingOutput* moved_ptr = static_cast<CouplingOutput*>(this->coupling_output[moved_out_idx].root());
            if (moved_ptr != nullptr)
            {
                *moved_ptr = static_cast<CouplingOutput>(moved);
            }
        }

        if (is_ping_pong)
        {
            active_buffer_idx_ = 1 - active_buffer_idx_;
        }

        return this->coupling_output;
    }

private:
    void init_internal_buffers()
    {
        if (this->coupling_input.size() >= 2)
        {
            const auto& dims = this->coupling_input[0].dimensions();
            if (dims.size() >= 2)
            {
                local_nz_ = dims[0] - 2;
                local_nx_ = dims[1] - 2;
                pitch_ = dims[1];
                batch_size_ = static_cast<std::size_t>(std::max(0, local_nz_)) * static_cast<std::size_t>(std::max(0, local_nx_));
            }
        }

        const std::size_t chunk_cap = std::max<std::size_t>(1, batch_size_);
        model_output_buffer_.assign(chunk_cap, static_cast<LibraryOutput>(0));

        this->library_output = MLCouplingData<LibraryOutput>();
        this->library_output.add_tensor(
            MLCouplingTensor<LibraryOutput>::wrap_flat(
                model_output_buffer_.data(),
                std::vector<int>{static_cast<int>(chunk_cap)},
                MLCouplingMemLayoutContiguous,
                MLCouplingOwnershipExternal));

        const bool use_flat = (model_io_layout_ == "flat_contiguous");
        this->library_input = MLCouplingData<LibraryInput>();

        if (use_flat)
        {
            flat_input_buffer_.assign(chunk_cap * 18, static_cast<LibraryInput>(0));
            this->library_input.add_tensor(
                MLCouplingTensor<LibraryInput>::wrap_flat(
                    flat_input_buffer_.data(),
                    std::vector<int>{static_cast<int>(chunk_cap), 18},
                    MLCouplingMemLayoutContiguous,
                    MLCouplingOwnershipExternal));
        }
        else
        {
            water_batch_.resize(chunk_cap);
            water_channels_.resize(chunk_cap);
            water_rows_.resize(chunk_cap * 3, nullptr);

            terrain_batch_.resize(chunk_cap);
            terrain_channels_.resize(chunk_cap);
            terrain_rows_.resize(chunk_cap * 3, nullptr);

            for (std::size_t k = 0; k < chunk_cap; ++k)
            {
                water_batch_[k] = &water_channels_[k];
                water_channels_[k] = &water_rows_[k * 3];

                terrain_batch_[k] = &terrain_channels_[k];
                terrain_channels_[k] = &terrain_rows_[k * 3];
            }

            this->library_input.add_tensor(
                MLCouplingTensor<LibraryInput>::wrap_nested(
                    static_cast<void*>(water_batch_.data()),
                    std::vector<int>{static_cast<int>(chunk_cap), 1, 3, 3},
                    MLCouplingMemLayoutNested,
                    MLCouplingOwnershipExternal));
            this->library_input.add_tensor(
                MLCouplingTensor<LibraryInput>::wrap_nested(
                    static_cast<void*>(terrain_batch_.data()),
                    std::vector<int>{static_cast<int>(chunk_cap), 1, 3, 3},
                    MLCouplingMemLayoutNested,
                    MLCouplingOwnershipExternal));
        }
    }

    std::string model_io_layout_ = "split_3x3";
    int active_buffer_idx_ = 0;
    int local_nz_ = 0;
    int local_nx_ = 0;
    int pitch_ = 0;
    std::size_t batch_size_ = 0;
    double last_moved_ = 0.0;

    std::vector<LibraryInput> flat_input_buffer_;
    std::vector<LibraryInput***> water_batch_;
    std::vector<LibraryInput**> water_channels_;
    std::vector<LibraryInput*> water_rows_;
    std::vector<LibraryInput***> terrain_batch_;
    std::vector<LibraryInput**> terrain_channels_;
    std::vector<LibraryInput*> terrain_rows_;
    std::vector<LibraryOutput> model_output_buffer_;
};
