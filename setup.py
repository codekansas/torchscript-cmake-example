#!/usr/bin/env python

import multiprocessing
import os
import platform
import shutil
import subprocess
import sys

import torch._C
import torch.cuda
import torch.utils
import torch.utils.cpp_extension
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name: str) -> None:
        super().__init__(name, sources=[])
        self.source_path = os.path.abspath(name)
        self.name = name


class CMakeBuild(build_ext):
    def run(self) -> None:
        if not shutil.which("cmake") or torch.utils.cmake_prefix_path is None:
            raise RuntimeError(f"CMake prefix path not found")

        # Sets paths to various CMake stuff.
        cmake_prefix_paths = [torch.utils.cmake_prefix_path]
        self.cmake_prefix_path = ";".join(cmake_prefix_paths)
        self.python_path = shutil.which("python")

        for ext in self.extensions:
            assert isinstance(ext, CMakeExtension)
            self.build_cmake(ext)

    def build_cmake(self, ext: CMakeExtension) -> None:
        output_path = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={output_path}",
            f"-DCMAKE_PREFIX_PATH={self.cmake_prefix_path}",
            f"-DPYTHON_EXECUTABLE:FILEPATH={self.python_path}",
        ]

        config = "Debug" if self.debug else "Release"
        build_args = ["--config", config]

        if platform.system() == "Darwin":
            cmake_args += ["-DCMAKE_OSX_DEPLOYMENT_TARGET=10.9"]

        if platform.system() == "Windows":
            cmake_args += [f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{config.upper()}={output_path}"]
            if sys.maxsize > 2 ** 32:
                cmake_args += ["-A", "x64"]
            build_args += ["--", "/m"]
        else:
            cmake_args += [f"-DCMAKE_BUILD_TYPE={config}"]

        env = os.environ.copy()

        # Builds CMake to a temp directory.
        build_temp = os.path.abspath(self.build_temp)
        if not os.path.exists(build_temp):
            os.makedirs(build_temp)
        subprocess.check_call(["cmake", f"-S{ext.source_path}", f"-B{build_temp}"] + cmake_args, env=env)

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
