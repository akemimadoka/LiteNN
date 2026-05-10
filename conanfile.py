from conan import ConanFile
from conan.errors import ConanInvalidConfiguration
from conan.tools.cmake import CMake, CMakeToolchain, cmake_layout
from conan.tools.files import copy, load

import os
import re


class LiteNNConan(ConanFile):
    name = "litenn"
    package_type = "library"
    license = "MIT"
    description = "LiteNN is a C++26 compiler-style neural network library."
    topics = ("neural-network", "mlir", "compiler", "cxx26")

    settings = "os", "arch", "compiler", "build_type"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "with_mlir": [True, False],
        "with_sanitizers": [True, False],
    }
    default_options = {
        "shared": False,
        "fPIC": True,
        "with_mlir": False,
        "with_sanitizers": False,
    }

    def set_version(self):
        cmake_lists = load(self, os.path.join(self.recipe_folder, "CMakeLists.txt"))
        match = re.search(r"project\(LiteNN VERSION ([^\s\)]+)", cmake_lists)
        if not match:
            raise ConanInvalidConfiguration("Unable to infer LiteNN version from CMakeLists.txt")
        self.version = match.group(1)

    def export_sources(self):
        for file_name in ("CMakeLists.txt", "LICENSE", "README.md"):
            copy(self, file_name, self.recipe_folder, self.export_sources_folder)

        for directory in (
            "cmake",
            os.path.join("src"),
            os.path.join("third_party", "googletest"),
        ):
            copy(self, "*", os.path.join(self.recipe_folder, directory), os.path.join(self.export_sources_folder, directory))

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def configure(self):
        if self.options.shared:
            self.options.rm_safe("fPIC")

    def layout(self):
        cmake_layout(self)

    def validate(self):
        compiler = str(self.settings.compiler)
        if compiler not in ("gcc", "clang"):
            raise ConanInvalidConfiguration(
                "LiteNN currently requires a reflection-capable GCC/Clang toolchain. "
                "The default MSVC Conan profile is not supported by this codebase."
            )

        cppstd = self.settings.compiler.get_safe("cppstd")
        if cppstd:
            normalized_cppstd = str(cppstd)
            if normalized_cppstd.startswith("gnu"):
                normalized_cppstd = normalized_cppstd[3:]
            try:
                cppstd_value = int(normalized_cppstd)
            except ValueError as exc:
                raise ConanInvalidConfiguration(
                    f"Unsupported cppstd setting '{cppstd}'. Use 26 or gnu26 for LiteNN."
                ) from exc
            if cppstd_value < 26:
                raise ConanInvalidConfiguration(
                    "LiteNN requires compiler.cppstd=26 or gnu26. "
                    "Example: conan create . -s compiler=gcc -s compiler.version=<version> -s compiler.cppstd=gnu26"
                )

    def generate(self):
        toolchain = CMakeToolchain(self)
        toolchain.cache_variables["BUILD_TESTING"] = False
        toolchain.cache_variables["LITENN_BUILD_EXAMPLES"] = False
        toolchain.cache_variables["LITENN_BUILD_BENCHMARKS"] = False
        toolchain.cache_variables["LITENN_ENABLE_MLIR"] = bool(self.options.with_mlir)
        toolchain.cache_variables["LITENN_ENABLE_SANITIZERS"] = bool(self.options.with_sanitizers)
        toolchain.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build(target="LiteNN")
        if self.options.with_mlir:
            cmake.build(target="LiteNNCompiler")

    def package(self):
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        self.cpp_info.libs = ["LiteNN"]
        self.cpp_info.builddirs = [os.path.join("lib", "cmake", "LiteNN")]
        self.cpp_info.set_property("cmake_find_mode", "none")