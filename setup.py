from setuptools import setup, Extension
import pybind11
import platform

# Get include directories for pybind11
include_dirs = [pybind11.get_include()]
system = platform.system()

libraries = ["armadillo"]
extra_link_args = []
extra_compile_args = ["-std=c++17"]
if system == "Windows":
    extra_compile_args += ["/openmp"]
elif system == "Darwin":
    # macOS needs -Xpreprocessor -fopenmp, plus link against -lomp
    extra_compile_args += ["-Xpreprocessor", "-fopenmp"]
    extra_link_args += ["-lomp"]
else:
    # Linux
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

