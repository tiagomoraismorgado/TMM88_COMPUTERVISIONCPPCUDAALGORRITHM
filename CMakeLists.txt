cmake_minimum_required(VERSION 3.24)

project(cuda_mad_exploits
    VERSION 0.1.0
    DESCRIPTION "GPU-accelerated Most Apparent Distortion (MAD) quality metric"
    HOMEPAGE_URL "https://github.com/yourname/cuda_mad_exploits"
    LANGUAGES CXX
)

# Optional CUDA support: found only if explicitly requested and available
if(ENABLE_CUDA)
    find_package(CUDAToolkit QUIET COMPONENTS cufft)
    if(CUDAToolkit_FOUND)
        set(HAVE_CUDA ON)
        message(STATUS "CUDA toolkit found")
    else()
        set(HAVE_CUDA OFF)
        message(WARNING "CUDA toolkit not found, disabling CUDA support")
    endif()
else()
    set(HAVE_CUDA OFF)
endif()


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS ON)

option(ENABLE_CUDA "Enable CUDA support" ON)
option(ENABLE_OPENCV "Enable OpenCV support" ON)
option(ENABLE_WARNINGS "Enable extra compiler warnings" ON)
option(ENABLE_WERROR "Treat warnings as errors" OFF)
option(ENABLE_CLANG_FORMAT "Enable clang-format target" ON)

# -----------------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------------
find_package(OpenCV 4.8 QUIET COMPONENTS core imgproc imgcodecs)
if(OpenCV_FOUND AND ENABLE_OPENCV)
    set(HAVE_OPENCV ON)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
else()
    set(HAVE_OPENCV OFF)
    message(WARNING "OpenCV not found or disabled (ENABLE_OPENCV=${ENABLE_OPENCV}).")
endif()

# -----------------------------------------------------------------------------------
# Source configuration
# -----------------------------------------------------------------------------------
set(SOURCE_FILES main.cpp)
if(HAVE_CUDA AND HAVE_OPENCV)
    list(APPEND SOURCE_FILES kernel.cu)
endif()

add_executable(cuda_mad_exploits ${SOURCE_FILES})

target_compile_definitions(cuda_mad_exploits
    PRIVATE
        $<$<BOOL:${HAVE_CUDA}>:HAVE_CUDA>
        $<$<BOOL:${HAVE_OPENCV}>:HAVE_OPENCV>
)

target_include_directories(cuda_mad_exploits
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        $<$<BOOL:${HAVE_OPENCV}>:${OpenCV_INCLUDE_DIRS}>
)

target_link_libraries(cuda_mad_exploits
    PRIVATE
        $<$<BOOL:${HAVE_OPENCV}>:${OpenCV_LIBS}>
        $<$<BOOL:${HAVE_CUDA}>:CUDA::cufft>
)

if(ENABLE_WARNINGS)
    if(MSVC)
        target_compile_options(cuda_mad_exploits PRIVATE /W4 /permissive-)
    else()
        target_compile_options(cuda_mad_exploits PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    endif()
endif()

if(ENABLE_WERROR)
    if(MSVC)
        target_compile_options(cuda_mad_exploits PRIVATE /WX)
    else()
        target_compile_options(cuda_mad_exploits PRIVATE -Werror)
    endif()
endif()

if(ENABLE_CLANG_FORMAT)
    find_program(CLANG_FORMAT_EXE clang-format)
    if(CLANG_FORMAT_EXE)
        file(GLOB_RECURSE FORMAT_SOURCES
            "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
            "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
        )
        add_custom_target(format
            COMMAND ${CLANG_FORMAT_EXE} -i ${FORMAT_SOURCES}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Running clang-format"
        )
    endif()
endif()

# Testing
enable_testing()
add_test(NAME help_command COMMAND cuda_mad_exploits --help)

# Unit test target
add_subdirectory(tests)

# Sanity target
add_custom_target(sanity
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target cuda_mad_exploits
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Run sanity build + tests"
)

# Compatibility helper message
if(NOT HAVE_CUDA OR NOT HAVE_OPENCV)
    message(STATUS "Building fallback binary (no CUDA/OpenCV). Executable will run in basic mode.")
endif()
