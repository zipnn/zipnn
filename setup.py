from setuptools import setup, find_packages, Extension
import subprocess
import os
import platform

def update_submodules():
    if os.path.exists(".git"):
        try:
            subprocess.check_call(["git", "submodule", "update", "--init", "--recursive"])
        except subprocess.CalledProcessError as e:
            print(f"Failed to update submodules: {e}")
            raise

update_submodules()

# Collect all zstd source files
zstd_sources = []
for root, _, files in os.walk('include/zstd/lib'):
    for file in files:
        if file.endswith('.c'):
            zstd_sources.append(os.path.join(root, file))

# Add your project-specific source files
zstd_sources.extend([
    "csrc/split_dtype_module.c",
    "csrc/split_dtype.c",
    "csrc/data_manipulation_dtype16.c",
    "csrc/data_manipulation_dtype32.c",
    "csrc/methods_enums.c",
])

# Determine the appropriate compiler and linker flags
extra_compile_args = [
    "-O3", "-Wall", "-Wextra",
    "-DZSTD_MULTITHREAD",
    "-DZSTD_DISABLE_ASM",  # Disable assembly optimizations
#    "-DZSTD_NO_INTRINSICS"  # Disable intrinsics
]
extra_link_args = []

if platform.system() == "Linux":
    extra_compile_args.extend(["-pthread", "-march=native"])
    extra_link_args.extend(["-pthread", "-lm"])

split_dtype_extension = Extension(
    "split_dtype",
    sources=zstd_sources,
    include_dirs=["include/zstd/lib/", "include/zstd/lib/common", "csrc/"],
    define_macros=[
        ('ZSTD_MULTITHREAD', None),
        ('ZSTD_DISABLE_ASM', None),
#        ('ZSTD_NO_INTRINSICS', None)
    ],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

setup(
    name="zipnn",
    version="0.3.7",
    author="ZipNN Contributors",
    author_email="moshik1@gmail.com",
    description="A lossless and near-lossless compression method optimized for numbers/tensors in the Foundation Models environment",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/zipnn/zipnn",
    packages=find_packages(include=["zipnn", "zipnn.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "zstandard",
        "torch",
    ],
    ext_modules=[split_dtype_extension],
)
