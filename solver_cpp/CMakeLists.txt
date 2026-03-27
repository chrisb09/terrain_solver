# Definitions
set(PROJECT_NAME "terrain_solver")
set(MAIN_CPP_FILE "terrain_solver.cpp")
set(SMARTREDIS_DIR "/hpcwork/ro092286/smartsim/SmartRedis")
set(CUSTOM_CORE_DEPENDENCIES_DIR "/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software")




## CMake configuration for the terrain_solver mini-app
cmake_minimum_required(VERSION 3.14)
project(${PROJECT_NAME} LANGUAGES CXX)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Use C++17 (no compiler extensions)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Require the MPI C++ bindings for parallel execution
find_package(MPI REQUIRED COMPONENTS CXX)

# Prefer a parallel HDF5 build; we use the C interface here
set(HDF5_PREFER_PARALLEL TRUE)
find_package(HDF5 REQUIRED COMPONENTS C)

# Warn if HDF5 is not built with parallel support. The application
# expects parallel HDF5 when run with --io-mode parallel_hdf5.
if(HDF5_C_IS_PARALLEL OR HDF5_IS_PARALLEL)
    message(STATUS "HDF5 parallel support: YES")
else()
    message(WARNING "HDF5 was found but is NOT built with parallel support. "
                    "--io-mode parallel_hdf5 will fail at runtime. "
                    "Load a parallel HDF5 module (e.g. HDF5/1.14.3 built against OpenMPI/4.1.6).")
endif()


# Build the executable and link with MPI + HDF5
add_executable(${PROJECT_NAME} ${MAIN_CPP_FILE})
target_include_directories(${PROJECT_NAME} PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE MPI::MPI_CXX ${HDF5_LIBRARIES})

# Manually set SmartRedis include and library paths
target_include_directories(${PROJECT_NAME} PRIVATE ${SMARTREDIS_DIR}/include)
target_include_directories(${PROJECT_NAME} PRIVATE ${SMARTREDIS_DIR}/install/include)
target_link_directories(${PROJECT_NAME} PRIVATE ${SMARTREDIS_DIR}/install/lib64)
target_link_libraries(${PROJECT_NAME} PRIVATE smartredis)

# Add Intel compiler libraries path for SmartRedis dependencies
target_link_directories(${PROJECT_NAME} PRIVATE ${CUSTOM_CORE_DEPENDENCIES_DIR}/imkl/2024.2.0/compiler/2024.2/lib)

# Link against GCC's libstdc++
target_link_directories(${PROJECT_NAME} PRIVATE ${CUSTOM_CORE_DEPENDENCIES_DIR}/GCCcore/11.3.0/lib64)
