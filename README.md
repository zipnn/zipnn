# ZipNN

## Introduction

ZipNN is a lossless and near-lossless compression method optimized for numbers/tensors in the Foundation Models environment. 
It adds Bit-Manipulation before and after vanilla compression/decompression methods.

 - [Byte Grouping](./docs/BITMANIPULATION.md#byte-grouping) - grouping together similar bytes (group together the first byte from all parameters, then the second byte, etc.) - The default is ByteGroup of 4 (partitions to 4 groups)
 - [Sign Bit](./docs/BITMANIPULATION.md#signbit-handeling) -  (On the way) - Move the sign bit since it hold high entropy. 
 - [Tunable Lossy Compression](./docs/BITMANIPULATION.md#tunable-lossy-compression) -This technique allows for incurring controlled inaccuracies to parameters, under the assumption that a lot of the entropy in model weights is actually redundant, i.e., noise saved to disk.
 - [Delta](./docs/BITMANIPULATION.md#delta-compression) - (On the way) - Compute the difference between the two inputs (for exmple: models)

For more details, please see our paper: [Lossless and Near-Lossless Compression for Foundation Models](https://arxiv.org/pdf/2404.15198)

Currently, ZipNN compression methods are implemented on CPUs, and GPU implementations are on the way. 

## zipnn package

Zipnn is a tool designed to compress and decompress data in byte, file, and Torch tensor formats. This repository includes implementations for compressing data into byte or file formats and decompressing it back to byte, file, or Torch tensor formats. The Zipnn package implements support for several kinds of compression.

<p align="center">
  <img src="./images/updated_flow.png" alt="Flow Image" width="800" height="400" style="display: block; margin: 0 auto;">
</p>


## Installation

Install with pip:

```sh
pip install zipnn
```

### For specific Compression methods other than ZSTD

* For lz4 method: ```pip install lz4```
* For snappy method: ```pip install python-snappy```


### For compressing/decompressing PyTorch tensor:

```
pip install torch
```

## Usage

Import zipnn

```python
from zipnn import zipnn
```

Instance class:

```python
zipnn = zipnn.ZipNN(method='zstd')
```

Compression:

```python
compressed_data = zipnn.compress(example_string)
```

Decompression:

```python
decompressed_data = zipnn.decompress(compressed_data)
```

## Example

```python
from zipnn import zipnn

example_string = b"Example string for compression"

# Initializing the ZipNN class with the default configuration
# for Byte->Byte compression and Byte->Byte decompression
zipnn = zipnn.ZipNN(method='zstd')

# Compress the byte string
compressed_data = zipnn.compress(example_string)

# Decompress the byte string back
decompressed_data = zipnn.decompress(compressed_data)

# Verify the result
print("Are the original and decompressed byte strings the same? ", example_string == decompressed_data)
>>> True

```

## Configuration

The default configuration is ByteGrouping of 4 with vanilla ZSTD (running with 8 threads), and the input and outputs are "byte"
For more advanced methods, please see the following option:

* ```method```: Compression method, Supporting zstd, lz4, snappy (default value = zstd).
* ```delta_compressed_type```: Type of delta compression if chosen (default value = None, supports byte and file).
* ```bg_partitions```: Number of partitions for Byte Grouping (default value = 4).
* ```bg_compression_threshold```: Compression threshold of Byte Grouping (default value = 0.99).
* ```torch_dtype```: If a non-torch compressed file is decompressed as torch, it's dtype should be noted (default value = None).
* ```torch_shape```: If a non-torch compressed file is decompressed as torch, it's shape should be noted (default value = None).
* ```signbit_to_lsb```: Flag for moving the sign bit to the lsb to have all the exponent byte together in FP32 and BF16, only supported with lossy compression (default value = False).
* ```lossy_compressed_type```: Type for lossy compression if wanted, supporting only integer (default value = None).
* ```lossy_compressed_factor```: Compression factor for lossy compression (default value = 27).
* ```is_streaming```: Streaming flag (default value = False, supports only file at the moment).
* ```streaming_chunk_KB```: Chunk size for streaming if is_streaming is True (default value = 1MB).
* ```input_type```: Supporting byte, torch, file (default value = byte, and in case of file, enter the file name).
* ```input_file```: Path to the input file (default value = byte, and in case of file, enter none).
* ```compressed_ret_type```: The Compression type, Supporting byte, file (default value = byte).
* ```compressed_file```: Path to the compressed file, if compress_ret_type is file.
* ```decompressed_ret_type```: The Decompression type, Supporting byte, torch, file (default value = byte).
* ```decompressed_file```: Path to the decompressed file.
* ```zstd_level```: Compression level for zstd (default value = 3).
* ```zstd_threads```: Number of threads to be used for zstd compression (default value = 8).
* ```lz4_compression_level```: Compression level for lz4 (default value = 0).


### Validation test

Run tests for Byte/File input types, Byte/File compression types, Byte/File decompression types.


```sh
python3 -m unittest discover -s tests/ -p test_suit.py
```

## Support and Questions
We are excited to hear your feedback!

For issues and feature requests, please open a GitHub issue.

## Contributing
We welcome and value all contributions to the project!
