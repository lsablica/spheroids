from setuptools import setup, Extension, find_packages
import pybind11
import os

# Find Eigen headers
eigen_include = "/usr/include/eigen3"  # Default location on Ubuntu/Debian
if not os.path.exists(eigen_include):
    raise RuntimeError("Eigen headers not found. Please install libeigen3-dev")

ext_modules = [
    Extension(
        "spheroids.cpp._pkbd",
        ["src/spheroids/cpp/_pkbd.cpp"],
        include_dirs=[
            pybind11.get_include(),
            eigen_include
        ],
        language='c++',
        extra_compile_args=['-std=c++14']
    ),
    Extension(
        "spheroids.cpp._scauchy",
        ["src/spheroids/cpp/_scauchy.cpp"],
        include_dirs=[
            pybind11.get_include(),
            eigen_include
        ],
        language='c++',
        extra_compile_args=['-std=c++14']
    ),
    Extension(
        "spheroids.cpp._rpkbd",
        ["src/spheroids/cpp/_rpkbd.cpp"],
        include_dirs=[
            pybind11.get_include(),
            eigen_include
        ],
        language='c++',
        extra_compile_args=['-std=c++14']
    )
]

setup(
    name="spheroids",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=ext_modules,
    zip_safe=False,
    python_requires=">=3.8",
)
