# ZipNN - A holistic solution to reduce storage and transfer needs for all AI pipelines

## Introduction

In the realm of data compression, achieving a high compression/decompression ratio often requires careful consideration of the data types and the nature of the datasets being compressed. For instance, different strategies may be optimal for floating-point numbers compared to integers, and datasets in monotonic order may benefit from distinct preparations.

ZipNN is a lossless and near-lossless compression method optimized for numbers/tensors in the Foundation Models environment, designed to automatically prepare the data for compression according to its type. By simply calling zipnn.compress(data), users can rely on the package to apply the most effective compression technique under the hood.

[Click here to explore the options we use for different datasets and data types](./UTH.md)

With zipnn, users can focus on their core tasks without worrying about the complexities of data compression, confident that the package will deliver the best possible results for their specific data types and structures.

For more details, please see our paper: [Lossless and Near-Lossless Compression for Foundation Models](https://arxiv.org/pdf/2404.15198)

Currently, ZipNN compression methods are implemented on CPUs, and GPU implementations are on the way. 

Given a specific data set, ZipNN Automatically rearranges the data according to it's type, and applies the most effective techniques for the given instance to improve compression ratios and rates.

<p align="center">
  <img src="./images/updated_flow.png" alt="Flow Image" width="900" height="390" style="display: block; margin: 0 auto;">
</p>

## Results

Below is a comparison of compression results between ZipNN and several other methods on bfloat16 data.

| Compressor name | Compression ratio / Output size | Compression Throughput | Decompression Throughput |
|-----------|--------------------------------|------------------------|--------------------------|
| ZipNN     | 1.51 / 66.3%                 | 1120MB/sec          | 1660MB/sec            |
| ZSTD      | 1.27 / 78.3%                 | 785MB/sec           | 950MB/sec             |
| LZ4       | 1 / 100%                     | ---                    | ---                      |
| Snappy    | 1 / 100%                     | ---                    | ---                      |


* Gzip, Zlib compression rate are similar to ZSTD, but much slower.
* The above results are for a single-threaded compression (Working with chunks size of 256KB).
* Similar results with other BF16 Models such as Mistral, Lamma-3, Lamma-3.1, Arcee-Nova and Jamba.

## Installation
Install using pip:

```sh
pip install zipnn
```

We are using two submodules:
* Cyan4973/FiniteStateEntropy [https://github.com/Cyan4973/FiniteStateEntropy]
* facebok/zstd [https://github.com/facebook/zstd] tag 1.5.6

### Dependencies

This project requires the following Python packages:

* numpy
* zstandard
* torch

### For specific Compression methods other than ZSTD

* For lz4 method: ```pip install lz4```
* For snappy method: ```pip install python-snappy```

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

We've included an example that compresses and decompresses 1GB of the Granite model and validates that the original file and the decompressed file are equal.

```
>>> python3 simple_example_granite.py
>>> Are the original and decompressed byte strings the same [BYTE]?  True
```

## Configuration

The default configuration is ByteGrouping of 4 with vanilla ZSTD (running with 8 threads), and the input and outputs are "byte". For more advanced options, please consider the following parameters:

* ```method```: Compression method, Supporting zstd, lz4, snappy (default value = 'zstd').
* ```input_format```: The input data format, can be one of the following: torch, numpy, byte (default value = 'byte').
* ```bytearray_dtype```: The data type of the byte array, if input_format is 'byte'. If input_format is torch or numpy, the dtype will be derived from the data automatically (default value = 'float32').
* ```threads```: The maximum threads for the compression and the bit manipulation. If 0, the code decides according to the dataset length (default value = 1).
* ```compression_threshold```: Only relevant for a compression that uses byte grouping. Compression threshhold for the byte grouping (default value = 0.95).
* ```byte_reorder```: Number of grouping. The format is the following:
  - Bit Format:
    - `[7]` - Group 0/1: 4th Byte
    - `[6-5]` - Group 0/1/2: 3rd Byte
    - `[4-3]` - Group 0/1/2/3: 2nd Byte
    - `[2-0]` - Group 0/1/2/3/4: 1st Byte

  - Examples:
    - bg16: Two groups - `0_00_01_010` (decimal 10)
    - fp32: Four groups - `1_10_11_100` (decimal 220)
    - int32: Truncate two MSBs - `0_00_01_001` (decimal 9)

* ```reorder_signbit```: This parameter controls the reordering of the sign bit for float32 or bfloat16 to improve compression. Options are:
    - `255`: No reordering of the sign bit.
    - `16`: Reorders the sign bit for bfloat16.
    - `32`: Reorders the sign bit for float32.
    - `0`: Automatically decides based on the data type (default value = 0).
 
*  ```compression_chunk```: Chunk size for compression. (default value = 256KB).

For even more advanced parameters, please refer to the [Under the Hood file.](./UTH.md#Additional-ZipNN-Configuration)

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

