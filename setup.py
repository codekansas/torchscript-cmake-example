#!/usr/bin/env python

import multiprocessing
import os
import platform
import shutil
import subprocess
from typing import List

import torch._C
import torch.utils
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name: str) -> None:
        super().__init__(name, sources=[])
        self.source_path = os.path.abspath(name)
        self.name = name


class CMakeBuild(build_ext):
    def run(self) -> None:
        if not shutil.which("cmake"):
            raise RuntimeError("CMake installation not found")
        if torch.utils.cmake_prefix_path is None:
            raise RuntimeError("CMake prefix path not found")

        # Need to copy PyBind flags.
        cmake_cxx_flags: List[str] = []
        for name in ["COMPILER_TYPE", "STDLIB", "BUILD_ABI"]:
            val = getattr(torch._C, f"_PYBIND11_{name}")
            if val is not None:
                cmake_cxx_flags += [f'-DPYBIND11_{name}=\\"{val}\\"']

        # Sets paths to various CMake stuff.
        self.cmake_prefix_path = torch.utils.cmake_prefix_path
        self.cmake_cxx_flags = " ".join(cmake_cxx_flags)
        self.python_path = shutil.which("python")

        for ext in self.extensions:
            assert isinstance(ext, CMakeExtension)
            self.build_cmake(ext)

    def build_cmake(self, ext: CMakeExtension) -> None:
        assert platform.system() == "Linux", f"Not supported: {platform.system()=}"
        output_path = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        config = "Debug" if self.debug else "Release"

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={output_path}",
            f"-DCMAKE_PREFIX_PATH={self.cmake_prefix_path}",
            f"-DPYTHON_EXECUTABLE:FILEPATH={self.python_path}",
            f"-DCMAKE_BUILD_TYPE={config}",
            f"-DCMAKE_CXX_FLAGS='{self.cmake_cxx_flags}'",
        ]

        env = os.environ.copy()

        # Builds CMake to a temp directory.
        build_temp = os.path.abspath(self.build_temp)
        if not os.path.exists(build_temp):
            os.makedirs(build_temp)
        subprocess.check_call(
            ["cmake", f"-S{ext.source_path}", f"-B{build_temp}"] + cmake_args,
            env=env,
        )

        # Compiles the project.
        build_lib = os.path.abspath(self.build_lib)
        if not os.path.exists(build_lib):
            os.makedirs(build_lib)
        subprocess.check_call(
            [
                "cmake",
                "--build",
                build_temp,
                "--",
                f"-j{multiprocessing.cpu_count()}",
            ],
            cwd=build_lib,
            env=env,
        )


with open("README.md") as f:
    long_description = f.read()

setup(
    name=f"torchscript-example",
    version=f"0.0.1",
    description="TorchScript CMake Example",
    author="Benjamin Bolte",
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    ext_modules=[CMakeExtension("example")],
    cmdclass={"build_ext": CMakeBuild},  # type: ignore
    include_package_data=True,
)
