# Definitions

message(STATUS "Directory: ${CMAKE_CURRENT_SOURCE_DIR}")

set(PROJECT_NAME "terrain_solver")
set(MAIN_CPP_FILE "terrain_solver.cpp")
set(SMARTREDIS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../CPP-ML-Interface/extern/SmartRedis")
set(CUSTOM_CORE_DEPENDENCIES_DIR "/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software")
set(CPPML_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../CPP-ML-Interface")
add_subdirectory("${CPPML_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/cpp-ml-interface-build" EXCLUDE_FROM_ALL)





# a fork with some debug prints
OPTION(USE_CUSTOM_SMARTREDIS "Use a custom SmartRedis build" OFF)
# pass this so we can use #ifdef in the code to print debug statements
if(USE_CUSTOM_SMARTREDIS)
    add_definitions(-DUSE_CUSTOM_SMARTREDIS)
    message(STATUS "Using custom SmartRedis build")
endif()

# option to use the CPP-ML-Interface instead of directly using SmartRedis in the mini-app
OPTION(USE_CPP_ML_INTERFACE "Use the CPP-ML-Interface instead of directly using SmartRedis in the mini-app" OFF)
if(USE_CPP_ML_INTERFACE)
    add_definitions(-DUSE_CPP_ML_INTERFACE)
    message(STATUS "Using CPP-ML-Interface in the mini-app")
endif()


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
if(USE_CPP_ML_INTERFACE)
    set(SMARTSIM_PYTHON "${CMAKE_CURRENT_SOURCE_DIR}/../../python/smartsim_cuda-12/bin/python"
        CACHE FILEPATH "Python interpreter with clang module available")

    if(NOT EXISTS "${SMARTSIM_PYTHON}")
        message(FATAL_ERROR "SMARTSIM_PYTHON not found: ${SMARTSIM_PYTHON}")
    endif()

    set(CPPML_GENERATED_REGISTRY "${CMAKE_CURRENT_BINARY_DIR}/generated_registry.hpp")
    set(TEMP_GENERATED_REGISTRY "${CMAKE_CURRENT_BINARY_DIR}/generated_registry.tmp.hpp")

    file(GLOB_RECURSE CPPML_HEADERS "${CPPML_DIR}/include/**/*.hpp")
    file(GLOB_RECURSE TERRAIN_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp")
    set(ALL_HEADERS ${CPPML_HEADERS} ${TERRAIN_HEADERS})
    list(REMOVE_DUPLICATES ALL_HEADERS)

    set(BASE_CLASSES "MLCouplingProvider,MLCouplingNormalization,MLCouplingBehavior,MLCouplingApplication")
    set(CLANG_ARGS "-std=c++17 -I${CPPML_DIR}/include -I${CMAKE_CURRENT_SOURCE_DIR}/include -I${SMARTREDIS_DIR}/include -I${SMARTREDIS_DIR}/install/include -xc++")

    add_custom_command(
        OUTPUT "${CPPML_GENERATED_REGISTRY}"
        COMMAND ${CMAKE_COMMAND} -E env "CLANG_ARGS=${CLANG_ARGS}"
                "${SMARTSIM_PYTHON}" "${CPPML_DIR}/scripts/generate_registry.py"
                "${TEMP_GENERATED_REGISTRY}" "${BASE_CLASSES}" ${ALL_HEADERS}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${TEMP_GENERATED_REGISTRY}" "${CPPML_GENERATED_REGISTRY}"
        DEPENDS ${ALL_HEADERS} "${CPPML_DIR}/scripts/generate_registry.py"
        WORKING_DIRECTORY "${CPPML_DIR}"
        COMMENT "Generating merged registry (CPP-ML-Interface + terrain_solver headers)..."
    )

    add_custom_target(terrain_solver_registry DEPENDS "${CPPML_GENERATED_REGISTRY}")
    add_dependencies(${PROJECT_NAME} terrain_solver_registry)
    target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_BINARY_DIR}")
endif()
target_include_directories(${PROJECT_NAME} PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE MPI::MPI_CXX ${HDF5_LIBRARIES})

# Manually set SmartRedis include and library paths
target_include_directories(${PROJECT_NAME} PRIVATE ${SMARTREDIS_DIR}/include)
target_include_directories(${PROJECT_NAME} PRIVATE ${SMARTREDIS_DIR}/install/include)
target_link_directories(${PROJECT_NAME} PRIVATE ${SMARTREDIS_DIR}/install/lib64)
target_link_libraries(${PROJECT_NAME} PRIVATE smartredis)

# Link against the CPP-ML-Interface headers (no .so required)
target_link_libraries(${PROJECT_NAME} PRIVATE cpp_ml_interface_headers)

# Add Intel compiler libraries path for SmartRedis dependencies
target_link_directories(${PROJECT_NAME} PRIVATE ${CUSTOM_CORE_DEPENDENCIES_DIR}/imkl/2024.2.0/compiler/2024.2/lib)

# Link against GCC's libstdc++
target_link_directories(${PROJECT_NAME} PRIVATE ${CUSTOM_CORE_DEPENDENCIES_DIR}/GCCcore/11.3.0/lib64)
