import sys
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

__version__ = "0.1.0"

cpp_sources = [
    "engine/src/block_allocator.cpp",
    "engine/src/sequence.cpp",
    "engine/src/scheduler.cpp",
    "engine/src/prefix_cache.cpp",
    "engine/src/bindings.cpp",
]

include_dirs = [
    "engine/include",
]

extra_compile_args = []
if sys.platform == "win32":
    extra_compile_args = ["/O2", "/std:c++17"]
else:
    extra_compile_args = ["-O3", "-std=c++17"]

ext_modules = [
    Pybind11Extension(
        "memforge_c",
        sources=cpp_sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        cxx_std=17,
    ),
]

setup(
    name="memforge",
    version=__version__,
    author="MemForge Team",
    description="High-performance custom LLM inference engine with PagedAttention and Speculative Decoding",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.9",
)
