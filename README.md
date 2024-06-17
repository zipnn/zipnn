# ZipNN

## Introduction

In the realm of data compression, achieving a high compression/decompression ratio often requires careful consideration of the data types and the nature of the datasets being compressed. For instance, different strategies may be optimal for floating-point numbers compared to integers, and datasets in monotonic order may benefit from distinct preparations.

ZipNN is a lossless and near-lossless compression method optimized for numbers/tensors in the Foundation Models environment, designed to automatically prepare the data for compression according to its type. By simply calling zipnn.compress(data), users can rely on the package to apply the most effective compression technique under the hood.

[Click here to explore the options we use for different datasets and data types](./UTH.md)

With zipnn, users can focus on their core tasks without worrying about the complexities of data compression, confident that the package will deliver the best possible results for their specific data types and structures.

For more details, please see our paper: [Lossless and Near-Lossless Compression for Foundation Models](https://arxiv.org/pdf/2404.15198)

Currently, ZipNN compression methods are implemented on CPUs, and GPU implementations are on the way. 

<p align="center">
  <img src="./updated_flow.png" alt="Flow Image" width="800" height="400" style="display: block; margin: 0 auto;">
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

There are 4 example files in the examples folder.

* To run a basic byte->byte compression & decompression:

```>>> python3.11 simple_example.py```

* To run byte->byte compression and 1 torch->torch compression on 3 huggingface models, and see the compression ratio:

```>>> python3.11 models_compression_ratio.py```

* To compress file->file run:

```>>> python3.11 cmd_comp.py input_file_name compressed_file_name```

* To decompress file->file run:

```>>> python3.11 cmd_decomp.py compressed_file_name decompressed_file_name```

## Configuration

The default configuration is ByteGrouping of 4 with vanilla ZSTD (running with 8 threads), and the input and outputs are "byte"
For more advanced methods, please see the following option:

* ```method```: Compression method, Supporting zstd, lz4, snappy (default value = 'zstd').
* ```input_format```: The input data format, can be one of the following: torch, numpy, byte (default value = 'byte').
* ```bytearray_dtype```: The data type of the byte array, if input_format is 'byte'. If input_format is torch or numpy, the dtype will be derived from the data automatically. (default value = 'float32').
* ```is_monotonic```: A boolean flag, set to True for a monotonic data. (default value = False).
* ```header_lean```: A boolean flag, set to True for a lean header. (default value = False).


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
