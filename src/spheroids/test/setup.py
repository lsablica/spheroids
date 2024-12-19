from setuptools import setup, Extension
import pybind11
import sys

# Include the pybind11 include directory
include_dirs = [pybind11.get_include()]

# If Armadillo is installed in a standard system location (like /usr/include), it should be found automatically.
# Otherwise, you might need to add that path to include_dirs.
# For example:
# include_dirs.append("/usr/include")


ext_modules = [
    Extension(
        "_pkbd",
        ["_pkbd.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["-std=c++14", "-DPYBIND11_DETAILED_ERROR_MESSAGES"],
        libraries=["armadillo"]
    )
]

setup(
    name="pkbdtest",
    version="0.1",
    ext_modules=ext_modules,
    install_requires=["pybind11", "numpy"],
)

