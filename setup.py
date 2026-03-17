from setuptools import setup, Extension
import pybind11
import platform
import os
import subprocess

system = platform.system()

include_dirs = [pybind11.get_include()]
libraries = []
extra_link_args = []
extra_compile_args = []

if system != "Windows":
    extra_compile_args.append("-std=c++17")

if system == "Windows":
    include_dirs.append(r"C:\vcpkg\installed\x64-windows\include")
    libraries = ["armadillo", "openblas", "lapack"]
    extra_compile_args += ["/std:c++17", "/openmp"]
    extra_link_args += [r"/LIBPATH:C:\vcpkg\installed\x64-windows\lib"]

elif system == "Darwin":
    # macOS: do NOT link to Homebrew's shared armadillo/libomp libs
    # Use Armadillo headers only + Accelerate
    arma_include = os.environ.get("ARMADILLO_INCLUDE_DIR")
    if not arma_include:
        def brew_prefix(pkg):
            return subprocess.check_output(["brew", "--prefix", pkg], text=True).strip()
        arma_include = os.path.join(brew_prefix("armadillo"), "include")

    include_dirs.append(arma_include)

    # Use Armadillo without its runtime wrapper library
    extra_compile_args += ["-DARMA_DONT_USE_WRAPPER"]

    # Link directly against Apple's BLAS/LAPACK implementation
    extra_link_args += ["-framework", "Accelerate"]

    # no libraries = ["armadillo"]
    # no OpenMP on macOS in wheel builds

else:
    # Linux
    libraries = ["armadillo"]
    extra_compile_args += ["-fopenmp"]

ext_modules = [
    Extension(
        "spheroids.cpp._estim",
        ["spheroids/cpp/_estim.cpp"],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        libraries=libraries,
        language="c++",
    ),
    Extension(
        "spheroids.cpp._utils",
        ["spheroids/cpp/_utils.cpp"],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        libraries=libraries,
        language="c++",
    ),
]

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="spheroids",
    version="0.4.0",
    author="Lukas Sablica",
    author_email="lsablica@wu.ac.at",
    description="A package for spherical clustering and probabilistic modeling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lsablica/spheroids",
    packages=["spheroids", "spheroids.cpp"],
    ext_modules=ext_modules,
    install_requires=[
        "numpy",
        "torch",
        "matplotlib",
    ],
    python_requires=">=3.8",
)
