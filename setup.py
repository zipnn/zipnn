from setuptools import setup, find_packages, Extension
import subprocess
import os


def update_submodules():
    if os.path.exists(".git"):
        try:
            subprocess.check_call(["git", "submodule", "update", "--init", "--recursive"])
        except subprocess.CalledProcessError as e:
            print(f"Failed to update submodules: {e}")
            raise


update_submodules()

split_dtype_extension = Extension(
    "split_dtype",
    sources=[
        "csrc/split_dtype_module.c",
        "csrc/split_dtype32.c",
        "csrc/split_dtype16.c",
        "include/FiniteStateEntropy/lib/fse_compress.c",
        "include/FiniteStateEntropy/lib/fse_decompress.c",
        "include/FiniteStateEntropy/lib/huf_compress.c",
        "include/FiniteStateEntropy/lib/huf_decompress.c",
        "include/FiniteStateEntropy/lib/entropy_common.c",
        "include/FiniteStateEntropy/lib/hist.c",
    ],
    include_dirs=["include/FiniteStateEntropy/lib/", "csrc/"],
    extra_compile_args=["-O3", "-Wall", "-Wextra"],
    extra_link_args=["-O3", "-Wall", "-Wextra"],
)

setup(
    name="zipnn",
    version="0.3.2",
    author="Moshik Hershcovitch",
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
    ext_modules=[split_dtype_extension],  # Add the C extension module here
)
