# Definitions

message(STATUS "Directory: ${CMAKE_CURRENT_SOURCE_DIR}")

cmake_minimum_required(VERSION 3.16)

# Mutually exclusive ML modes
option(USE_CPP_ML_INTERFACE "Use the CPP-ML-Interface in the mini-app" OFF)
option(WITH_DIRECT_SMARTSIM "Use direct SmartSim/SmartRedis in the mini-app (without CMI)" OFF)
option(WITH_DIRECT_AIX "Use direct AIxeleratorService in the mini-app (without CMI)" OFF)
option(USE_CUSTOM_SMARTREDIS "Use a custom SmartRedis build" OFF)

set(ACTIVE_ML_MODES 0)
if(USE_CPP_ML_INTERFACE)
    math(EXPR ACTIVE_ML_MODES "${ACTIVE_ML_MODES} + 1")
endif()
if(WITH_DIRECT_SMARTSIM)
    math(EXPR ACTIVE_ML_MODES "${ACTIVE_ML_MODES} + 1")
endif()
if(WITH_DIRECT_AIX)
    math(EXPR ACTIVE_ML_MODES "${ACTIVE_ML_MODES} + 1")
endif()

if(ACTIVE_ML_MODES GREATER 1)
    message(FATAL_ERROR "Mutually exclusive ML modes selected. Please enable at most one of: USE_CPP_ML_INTERFACE, WITH_DIRECT_SMARTSIM, WITH_DIRECT_AIX.")
endif()

if(USE_CPP_ML_INTERFACE)
    message(STATUS "Mode: USE_CPP_ML_INTERFACE (CPP-ML-Interface)")
    add_definitions(-DUSE_CPP_ML_INTERFACE)
elseif(WITH_DIRECT_SMARTSIM)
    message(STATUS "Mode: WITH_DIRECT_SMARTSIM (Direct SmartSim/SmartRedis)")
    add_definitions(-DWITH_DIRECT_SMARTSIM)
elseif(WITH_DIRECT_AIX)
    message(STATUS "Mode: WITH_DIRECT_AIX (Direct AIxeleratorService)")
    add_definitions(-DWITH_DIRECT_AIX)
else()
    message(STATUS "Mode: No-ML (Pure physics solver)")
endif()

if(USE_CUSTOM_SMARTREDIS)
    add_definitions(-DUSE_CUSTOM_SMARTREDIS)
    message(STATUS "Using custom SmartRedis build")
endif()

set(PROJECT_NAME "terrain_solver")
set(MAIN_CPP_FILE "terrain_solver.cpp")
set(SMARTREDIS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../CPP-ML-Interface/extern/SmartRedis")
set(CUSTOM_CORE_DEPENDENCIES_DIR "/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software")
set(CPPML_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../CPP-ML-Interface")
set(AIX_DIR "${CPPML_DIR}/extern/AIxeleratorService")

set(AIXELERATOR_PREBUILT_INSTALL_PREFIX "${AIX_DIR}/INSTALL" CACHE PATH "Install prefix for AIxeleratorService")

if(WITH_SCOREP OR (DEFINED ENV{USE_SCOREP} AND NOT "$ENV{USE_SCOREP}" STREQUAL "" AND NOT "$ENV{USE_SCOREP}" STREQUAL "0"))
    set(CMAKE_CXX_COMPILER scorep-mpicxx)
    set(CMAKE_C_COMPILER scorep-mpicc)
    set(CMAKE_CXX_COMPILER_ARG1 "--nocompiler --user")
    set(CMAKE_C_COMPILER_ARG1 "--nocompiler --user")
endif()

project(${PROJECT_NAME} LANGUAGES CXX C)

if(WITH_SCOREP OR (DEFINED ENV{USE_SCOREP} AND NOT "$ENV{USE_SCOREP}" STREQUAL "" AND NOT "$ENV{USE_SCOREP}" STREQUAL "0"))
    add_definitions(-DUSE_SCOREP)
    message(STATUS "Score-P support enabled for solver")
endif()

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

if(HDF5_C_IS_PARALLEL OR HDF5_IS_PARALLEL)
    message(STATUS "HDF5 parallel support: YES")
else()
    message(WARNING "HDF5 was found but is NOT built with parallel support. "
                    "--io-mode parallel_hdf5 will fail at runtime. "
                    "Load a parallel HDF5 module (e.g. HDF5/1.12.2 built against OpenMPI/4.1.4).")
endif()

# Build the executable and link with MPI + HDF5
add_executable(${PROJECT_NAME} ${MAIN_CPP_FILE})
target_include_directories(${PROJECT_NAME} PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE MPI::MPI_CXX ${HDF5_LIBRARIES})

# 1. CMI Mode
if(USE_CPP_ML_INTERFACE)
    option(WITH_PHYDLL "Enable PhyDLL provider support through CPP-ML-Interface" ON)
    set(CPPML_DISABLE_REGISTRY_GEN ON CACHE BOOL "Disable CPP-ML-Interface standalone registry generation" FORCE)
    add_subdirectory("${CPPML_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/cpp-ml-interface-build" EXCLUDE_FROM_ALL)

    set(SMARTSIM_PYTHON "${CMAKE_CURRENT_SOURCE_DIR}/../../python/smartsim_cuda-12/bin/python"
        CACHE FILEPATH "Python interpreter with clang module available")

    if(NOT EXISTS "${SMARTSIM_PYTHON}")
        message(FATAL_ERROR "SMARTSIM_PYTHON not found: ${SMARTSIM_PYTHON}")
    endif()

    set(CPPML_GENERATED_REGISTRY "${CMAKE_CURRENT_BINARY_DIR}/generated_registry.hpp")
    set(CPPML_LIBRARY_GENERATED_REGISTRY "${CMAKE_CURRENT_BINARY_DIR}/cpp-ml-interface-build/generated_registry.hpp")
    set(TEMP_GENERATED_REGISTRY "${CMAKE_CURRENT_BINARY_DIR}/generated_registry.tmp.hpp")

    file(GLOB_RECURSE CPPML_HEADERS "${CPPML_DIR}/include/**/*.hpp")
    file(GLOB_RECURSE TERRAIN_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp")
    set(ALL_HEADERS ${CPPML_HEADERS} ${TERRAIN_HEADERS})
    list(REMOVE_DUPLICATES ALL_HEADERS)

    set(BASE_CLASSES "MLCouplingLibrary,MLCouplingNormalization,MLCouplingBehavior,MLCouplingApplication")
    set(CLANG_ARGS "-std=c++17 -I${CPPML_DIR}/include -I${CMAKE_CURRENT_SOURCE_DIR}/include -I${SMARTREDIS_DIR}/include -I${SMARTREDIS_DIR}/install/include -xc++")

    add_custom_command(
        OUTPUT "${CPPML_GENERATED_REGISTRY}" "${CPPML_LIBRARY_GENERATED_REGISTRY}"
        COMMAND ${CMAKE_COMMAND} -E env "CLANG_ARGS=${CLANG_ARGS}"
                "${SMARTSIM_PYTHON}" "${CPPML_DIR}/scripts/generate_registry.py"
                "${TEMP_GENERATED_REGISTRY}" "${BASE_CLASSES}" ${ALL_HEADERS}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${TEMP_GENERATED_REGISTRY}" "${CPPML_GENERATED_REGISTRY}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${TEMP_GENERATED_REGISTRY}" "${CPPML_LIBRARY_GENERATED_REGISTRY}"
        DEPENDS ${ALL_HEADERS} "${CPPML_DIR}/scripts/generate_registry.py"
        WORKING_DIRECTORY "${CPPML_DIR}"
        COMMENT "Generating merged registry (CPP-ML-Interface + terrain_solver headers)..."
    )

    add_custom_target(terrain_solver_registry DEPENDS "${CPPML_GENERATED_REGISTRY}" "${CPPML_LIBRARY_GENERATED_REGISTRY}")
    add_dependencies(${PROJECT_NAME} terrain_solver_registry)
    add_dependencies(cpp_ml_interface_library terrain_solver_registry)
    target_include_directories(cpp_ml_interface_library PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_BINARY_DIR}")
    target_include_directories(${PROJECT_NAME} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_BINARY_DIR}"
        "${CPPML_DIR}/include")

    target_include_directories(${PROJECT_NAME} PRIVATE ${SMARTREDIS_DIR}/include ${SMARTREDIS_DIR}/install/include)
    target_link_directories(${PROJECT_NAME} PRIVATE ${SMARTREDIS_DIR}/install/lib64)
    target_link_libraries(${PROJECT_NAME} PRIVATE smartredis)
    target_link_libraries(${PROJECT_NAME} PRIVATE cpp_ml_interface_headers cpp_ml_interface_library)

# 2. Direct SmartSim Mode
elseif(WITH_DIRECT_SMARTSIM)
    target_include_directories(${PROJECT_NAME} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CPPML_DIR}/include"
        ${SMARTREDIS_DIR}/include
        ${SMARTREDIS_DIR}/install/include)
    target_link_directories(${PROJECT_NAME} PRIVATE ${SMARTREDIS_DIR}/install/lib64)
    target_link_libraries(${PROJECT_NAME} PRIVATE smartredis)

# 3. Direct AIx Mode
elseif(WITH_DIRECT_AIX)
    target_include_directories(${PROJECT_NAME} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${AIXELERATOR_PREBUILT_INSTALL_PREFIX}/include")
    target_link_directories(${PROJECT_NAME} PRIVATE
        "${AIXELERATOR_PREBUILT_INSTALL_PREFIX}/lib")
    target_link_libraries(${PROJECT_NAME} PRIVATE AIxeleratorService)
endif()

if(WITH_SCOREP OR (DEFINED ENV{USE_SCOREP} AND NOT "$ENV{USE_SCOREP}" STREQUAL "" AND NOT "$ENV{USE_SCOREP}" STREQUAL "0"))
    add_compile_options(--user)
    add_link_options(--user)
endif()

if(WITH_AIX OR USE_CPP_ML_INTERFACE)
    target_link_libraries(${PROJECT_NAME} PRIVATE nvidia-ml)
endif()
