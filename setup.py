from setuptools import setup, find_packages, Extension

# Define the C extension module, adjusting paths to source files
split_dtype_extension = Extension(
    'split_dtype',
    sources=[
        'csrc/split_dtype_module.c',
        'csrc/split_dtype32.c',
        'csrc/split_dtype16.c',
        'csrc/huf_api.c'
    ],
    include_dirs=[
        'csrc/FiniteStateEntropy/lib/',  
        'csrc/'  
    ],
    extra_compile_args=['-O3'],
    extra_link_args=['-O3']
)

setup(
    name='zipnn',
    version='0.1.1',
    author='Moshik Hershcovitch',
    author_email='moshik1@gmail.com',
    description='A lossless and near-lossless compression method optimized for numbers/tensors in the Foundation Models environment',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/zipnn/zipnn',
    packages=find_packages(include=['zipnn', 'zipnn.*']),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'zstandard',
        'torch',
    ],
    ext_modules=[split_dtype_extension]  # Add the C extension module here
)

