from setuptools import setup, Extension
import pybind11

include_dirs = [pybind11.get_include()]

ext_modules = [
    Extension(
        "_pkbd",
        ["_pkbd.cpp"],
        include_dirs=include_dirs,
        language="c++",
        extra_compile_args=["-std=c++14"],
        libraries=["armadillo"]
    ),
    Extension(
        "_rpkbd",
        ["_rpkbd.cpp"],
        include_dirs=include_dirs,
        language="c++",
        extra_compile_args=["-std=c++14"],
        libraries=["armadillo"]
    ),
    Extension(
        "_scauchy",
        ["_scauchy.cpp"],
        include_dirs=include_dirs,
        language="c++",
        extra_compile_args=["-std=c++14"],
        libraries=["armadillo"]
    ),
]

setup(
    name="my_project",
    version="0.1",
    ext_modules=ext_modules,
    install_requires=["pybind11", "numpy"],
)

