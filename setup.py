from setuptools import setup, Extension
import pybind11
import platform

# Get include directories for pybind11
system = platform.system()

libraries = ["armadillo"]
extra_link_args = []
extra_compile_args = []
if system != "Windows":
    extra_compile_args.append("-std=c++17")
    
if system == "Windows":
    extra_compile_args += ["/std:c++17", "/openmp"]
    # Tell the linker where to find .lib files from vcpkg
    extra_link_args += [
        "/LIBPATH:C:\\vcpkg\\installed\\x64-windows\\lib"
    ]
    libraries += ["openblas", "lapack"]
elif system == "Darwin":
    # macOS needs -Xpreprocessor -fopenmp, plus link against -lomp
    extra_compile_args += [
        "-Xpreprocessor", 
        "-fopenmp", 
        "-I/opt/homebrew/opt/libomp/include",
        "-I/opt/homebrew/opt/armadillo/include"
    ]
    extra_link_args += [
        "-lomp",
        "-L/opt/homebrew/opt/libomp/lib",
        "-L/opt/homebrew/opt/armadillo/lib"
    ]
else:
    # Linux
    extra_compile_args += ["-fopenmp"]

ext_modules = [
    Extension(
        "spheroids.cpp._estim",
        ["spheroids/cpp/_estim.cpp"],
        include_dirs=[pybind11.get_include(),
                      "C:\\vcpkg\\installed\\x64-windows\\include"
                     ] if system == "Windows" else [pybind11.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        libraries=libraries,
        language="c++",
    ),
    Extension(
        "spheroids.cpp._utils",
        ["spheroids/cpp/_utils.cpp"],
        include_dirs=[pybind11.get_include(),
                      "C:\\vcpkg\\installed\\x64-windows\\include"
                     ] if system == "Windows" else [pybind11.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        libraries=libraries,
        language="c++",
    ),
]

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()
# Define the setup configuration
setup(
    name="spheroids",
    version="0.2.0",
    author="Lukas Sablica",
    author_email="lsablica@wu.ac.at",
    description="A package for spherical clustering and probabilistic modeling",
    long_description=long_description,
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

