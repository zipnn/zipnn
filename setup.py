from setuptools import setup, find_packages, Extension
import subprocess
import os
import platform
import re

def rename_functions(content):
    # Skip renaming these specific types/defines
    skip_patterns = {
        'FSE_CTable',
        'FSE_DTable',
        'FSE_DTABLE_SIZE_U32',
    }

    # First handle specific patterns
    specific_patterns = [
        r'\bHUF_[a-zA-Z0-9_]+',
        r'\bFSE_[a-zA-Z0-9_]+',
        r'\bHIST_[a-zA-Z0-9_]+',
        r'\bg_debuglevel\b',
    ]

    for pattern in specific_patterns:
        matches = re.finditer(pattern, content)
        for match in matches:
            symbol = match.group(0)
            if not symbol.startswith('ZIPNN_') and symbol not in skip_patterns:
                content = re.sub(
                    r'\b' + re.escape(symbol) + r'\b',
                    'ZIPNN_' + symbol,
                    content
                )

    # Then handle prefixed functions
    prefixes_to_rename = [
        'FSE_',
        'HUF_',
        'HIST_',
        'DEBUG_',
        'ERR_'
    ]

    for prefix in prefixes_to_rename:
        pattern = r'\b' + prefix + r'[a-zA-Z0-9_]+'
        matches = re.finditer(pattern, content)
        for match in matches:
            symbol = match.group(0)
            if not symbol.startswith('ZIPNN_') and symbol not in skip_patterns:
                content = re.sub(
                    r'\b' + re.escape(symbol) + r'\b',
                    'ZIPNN_' + symbol,
                    content
                )

    return content

def prepare_fse_files():
    os.makedirs('build/modified_fse/lib', exist_ok=True)
    fse_src = 'include/FiniteStateEntropy/lib'
    
    fse_files = [
        'bitstream.h',
        'compiler.h',
        'debug.c',
        'debug.h',
        'entropy_common.c',
        'entropy_common.h',
        'error_private.h',
        'error_public.h',
        'fse.h',
        'fse_compress.c',
        'fse_decompress.c',
        'fseU16.c',
        'fseU16.h',
        'hist.c',
        'hist.h',
        'huf.h',
        'huf_compress.c',
        'huf_decompress.c',
        'mem.h',
    ]

    for file in fse_files:
        src = os.path.join(fse_src, file)
        dst = os.path.join('build/modified_fse/lib', file)

        if os.path.exists(src):
            with open(src, 'r', encoding='utf-8') as f:
                content = f.read()

            modified_content = rename_functions(content)

            with open(dst, 'w', encoding='utf-8') as f:
                f.write(modified_content)

def update_submodules():
    if os.path.exists(".git"):
        try:
            subprocess.check_call(["git", "submodule", "update", "--init", "--recursive"])
        except subprocess.CalledProcessError as e:
            print(f"Failed to update submodules: {e}")
            raise

# Run initialization steps
update_submodules()
prepare_fse_files()

# FSE source files
fse_sources = [
    "build/modified_fse/lib/huf_compress.c",
    "build/modified_fse/lib/huf_decompress.c",
    "build/modified_fse/lib/entropy_common.c",
    "build/modified_fse/lib/fse_compress.c",
    "build/modified_fse/lib/fse_decompress.c",
    "build/modified_fse/lib/hist.c",
    "build/modified_fse/lib/debug.c",
]

# Files to exclude from ZSTD compilation
zstd_exclude_files = {
#    'fse_compress.c',
#    'fse_decompress.c',
#    'huf_compress.c',
#    'huf_decompress.c',
#    'entropy_common.c',
#    'hist.c',
#    'debug.c'
}

# Collect ZSTD source files
zstd_sources = []
for root, _, files in os.walk('include/zstd/lib'):
    for file in files:
        if file.endswith('.c') and file not in zstd_exclude_files:
            zstd_sources.append(os.path.join(root, file))

# Project source files
project_sources = [
    "csrc/split_dtype_module.c",
    "csrc/split_dtype.c",
    "csrc/data_manipulation_dtype16.c",
    "csrc/data_manipulation_dtype32.c",
    "csrc/methods_utils.c",
]

# Compiler flags
extra_compile_args = [
    "-O3",
    "-Wall",
    "-Wextra",
    "-DXXH_PRIVATE_API",
    "-DZSTD_STATIC_LINKING_ONLY",
    "-DZIP_FSE_STATIC_LINKING_ONLY",
    "-DZIP_FSE_STATIC_BUILD",
]

if platform.system() == "Linux":
    extra_compile_args.extend(["-pthread", "-fPIC"])
    extra_link_args = ["-pthread", "-lm"]
else:
    extra_link_args = ["-lm"]

split_dtype_extension = Extension(
    "split_dtype",
    sources=project_sources + fse_sources + zstd_sources,
    include_dirs=[
        "build/modified_fse/lib/",
        "include/zstd/lib/",
        "include/zstd/lib/common",
        "include/zstd/lib/compress",
        "include/zstd/lib/decompress",
        "csrc/",
    ],
    define_macros=[
        ('ZSTD_STATIC_LINKING_ONLY', None),
        ('XXH_PRIVATE_API', None),
        ('ZIP_FSE_STATIC_LINKING_ONLY', None),
        ('ZIP_FSE_STATIC_BUILD', None),
    ],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

setup(
    name="zipnn",
    version="0.3.6",
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
