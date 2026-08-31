cmake_minimum_required(VERSION 3.18)

set(ENGINE_SOURCES
    src/block_allocator.cpp
    src/sequence.cpp
    src/scheduler.cpp
)

set(ENGINE_HEADERS
    include/memforge/types.hpp
    include/memforge/block_allocator.hpp
    include/memforge/sequence.hpp
    include/memforge/scheduler.hpp
    include/memforge/engine_core.hpp
)

# Core C++ static library
add_library(memforge_core STATIC
    ${ENGINE_SOURCES}
    ${ENGINE_HEADERS}
)

target_include_directories(memforge_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# pybind11 module if pybind11 is available
if(TARGET pybind11::module)
    pybind11_add_module(memforge_c
        src/bindings.cpp
        ${ENGINE_SOURCES}
    )
    target_include_directories(memforge_c PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(memforge_c PRIVATE pybind11::module)
endif()
