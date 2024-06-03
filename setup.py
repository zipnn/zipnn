from setuptools import setup, find_packages

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
        'License :: OSI Approved :: MIT License',  # Adjust based on your chosen license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',               
        'zstandard',         
    ],

)

