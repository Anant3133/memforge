cmake_minimum_required(VERSION 3.18)
project(memforge VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(MSVC)
    add_compile_options(/W4 /O2 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

# Find pybind11
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    # Attempt to locate pybind11 from python environment
    execute_process(
        COMMAND python -m pybind11 --cmake
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(pybind11_DIR)
        find_package(pybind11 CONFIG PATHS ${pybind11_DIR} NO_DEFAULT_PATH)
    endif()
endif()

# Include engine subdirectory
add_subdirectory(engine)

# Enable testing if GoogleTest is available or standalone
enable_testing()
add_subdirectory(tests/cpp)
