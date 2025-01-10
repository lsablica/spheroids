from setuptools import setup, Extension
import pybind11
import platform

# Get include directories for pybind11
include_dirs = [pybind11.get_include()]

import platform

if platform.system() == "Windows":
    extra_compile_args = ["/std:c++17", "/openmp"]
    libraries = ["armadillo"]  # Ensure Armadillo is available
else:
    extra_compile_args = ["-std=c++17", "-fopenmp"]
    libraries = ["armadillo"]


# Define the extensions
ext_modules = [
    Extension(
        "spheroids.cpp._estim",
        ["spheroids/cpp/_estim.cpp"],
        include_dirs=include_dirs,
        language="c++",
        extra_compile_args=extra_compile_args,  # Use C++17 and OpenMP
        libraries=["armadillo"],  # Link the Armadillo library
    ),
    Extension(
        "spheroids.cpp._utils",
        ["spheroids/cpp/_utils.cpp"],
        include_dirs=include_dirs,
        language="c++",
        extra_compile_args=extra_compile_args,  # Use C++17 and OpenMP
        libraries=["armadillo"],  # Link the Armadillo library
    ),
]

# Define the setup configuration
setup(
    name="spheroids",
    version="0.1.0",
    author="Lukas Sablica",
    author_email="lsablica@wu.ac.at",
    description="A package for spherical clustering and probabilistic modeling",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/lsablica/spheroids",
    packages=["spheroids", "spheroids.cpp"],
    ext_modules=ext_modules,
    install_requires=[
        "pybind11",
        "numpy",
        "torch",
        "matplotlib",
    ],
    python_requires=">=3.8",
)

