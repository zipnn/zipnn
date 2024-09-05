# ZipNN - A Lossless Compression Library for AI pipelines

**TL;DR - simple, fast and effective model compression**
## Getting started (fast)
Download the scripts for compressing/decompressing AI Models:

```
wget https://raw.githubusercontent.com/zipnn/zipnn/main/scripts/zipnn_compress_file.py
```
```
wget https://raw.githubusercontent.com/zipnn/zipnn/main/scripts/zipnn_decompress_file.py
```

To compress a file:
```
python3 zipnn_compress_file model_name
```

To decompress a file:
```
python3 zipnn_decompress_file compressed_model_name.znn
```

There are also scripts to compress/decompress all files in a folder.

## Introduction

In the realm of data compression, achieving a high compression/decompression ratio often requires careful consideration of the data types and the nature of the datasets being compressed. For instance, different strategies may be optimal for floating-point numbers compared to integers, and datasets in monotonic order may benefit from distinct preparations.

ZipNN is a lossless and near-lossless compression method optimized for numbers/tensors in the Foundation Models environment, designed to automatically prepare the data for compression according to its type. By simply calling zipnn.compress(data), users can rely on the package to apply the most effective compression technique under the hood.

[Click here to explore the options we use for different datasets and data types](./docs/UTH.md)

With zipnn, users can focus on their core tasks without worrying about the complexities of data compression, confident that the package will deliver the best possible results for their specific data types and structures.

For more details, please see our paper: [Lossless and Near-Lossless Compression for Foundation Models](https://arxiv.org/pdf/2404.15198)

Currently, ZipNN compression methods are implemented on CPUs, and GPU implementations are on the way. 

Given a specific data set, ZipNN Automatically rearranges the data according to it's type, and applies the most effective techniques for the given instance to improve compression ratios and rates.

<p align="center">
  <img src="./images/updated_flow.png" alt="Flow Image" style="display: block; margin: 0 auto;">
</p>

## Results

Below is a comparison of compression results between ZipNN and several other methods on bfloat16 data.

| Compressor name | Compression ratio / Output size | Compression Throughput | Decompression Throughput |
|-----------|--------------------------------|------------------------|--------------------------|
| ZipNN v0.2.0    | 1.51 / 66.3%                 | 1120MB/sec          | 1660MB/sec            |
| ZSTD v1.56     | 1.27 / 78.3%                 | 785MB/sec           | 950MB/sec             |
| LZ4       | 1 / 100%                     | ---                    | ---                      |
| Snappy    | 1 / 100%                     | ---                    | ---                      |


* Gzip, Zlib compression rate are similar to ZSTD, but much slower.
* The above results are for a single-threaded compression (Working with chunks size of 256KB).
* Similar results with other BF16 Models such as Mistral, Lamma-3, Lamma-3.1, Arcee-Nova and Jamba.

## Installation using pip

```sh
pip install zipnn
```

## Install source code

```
git clone git@github.com:zipnn/zipnn.git
cd zipnn
```

We are using two submodules:
* Cyan4973/FiniteStateEntropy [https://github.com/Cyan4973/FiniteStateEntropy]
* facebok/zstd [https://github.com/facebook/zstd] tag 1.5.6

```
git submodule update --init --recursive
```

Compile locally using pip
```
pip install -e .
```

### Dependencies

This project requires the following Python packages:

* numpy
* zstandard
* torch

### For specific Compression methods other than ZSTD

* For lz4 method: ```pip install lz4```
* For snappy method: ```pip install python-snappy```

## Usage

### Ready Made Scripts for file Compression/ Decompression

You can integrate zipnn compression and decompression into your own projects by utilizing the scripts available in the scripts folder. This folder contains the following scripts:
* ```zipnn_compress_file.py```: For compressing an individual file.
* ```zipnn_decompress_file.py```: For decompressing an individual file.
* ```zipnn_compress_path.py```: For compressing all files under a path.
* ```zipnn_decompress_path.py```: For decompressing all files under a path.

Compress one file:
```
python zipnn_compress_file.py model_name
```

Decompress one file:
```
python zipnn_decompress_file.py model_name.znn
```

For detailed information on how to use these scripts, [please refer to the README.md file located in the scripts folder](./scripts/README.md).


### Import Package Manually 

You can use the package manually, like so:

Import zipnn:

```python
from zipnn import ZipNN
```

Instance class:

```python
zpn = ZipNN(method='zstd', input_format='torch')
```

Create a 1MB tensor with random numbers from a uniform distribution between -1 and 1
The dtype is bfloat
```
import torch
original_tensor = torch.rand(10124*1024, dtype=torch.bfloat16) * 2 - 1
```

Compression:

```python
compressed_data = zpn.compress(original_tensor)
```

Decompression:

```python
decompressed_data = zpn.decompress(compressed_data)
```

Check for correctness:
```python
torch.equal(original_tensor, decompressed_data)
```

## Example

### Example of synthetic data
In this example, ZipNN compresses and decompresses 1MB of a random number between -1 to 1 in a torch.tensor format. 
```
> python3 simple_example.py
...
Are the original and decompressed byte strings the same [TORCH]?  True
``` 


### Example of a real module
In this example, ZipNN and ZSTD compress and decompress 1GB of the Granite model and validate that the original file and the decompressed file are equal. <br>
The script reads the file and compresses and decompresses in Byte format.

```
> python3 simple_example_granite.py
...
Are the original and decompressed byte strings the same [BYTE]?  True
```

## Configuration

The default configuration is ByteGrouping of 4 with vanilla ZSTD (running with 8 threads), and the input and outputs are "byte". For more advanced options, please consider the following parameters:

* ```method```: Compression method, Supporting zstd, lz4, snappy (default value = 'zstd').
* ```input_format```: The input data format, can be one of the following: torch, numpy, byte (default value = 'byte').
* ```bytearray_dtype```: The data type of the byte array, if input_format is 'byte'. If input_format is torch or numpy, the dtype will be derived from the data automatically (default value = 'float32').
* ```threads```: The maximum threads for the compression and the bit manipulation. If 0, the code decides according to the dataset length (default value = 1).
* ```compression_threshold```: Save original buffer if not compress above the threshold (default value = 0.95).
* ```check_th_after_percent```: Check the compression threshhold after % from the number of chunk and stop compressing if not pass the compression_threshold. (default value = 10[%]).
                 
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

[Click here to explore additional ZipNN configuration options](./docs/UTH.md#additional-zipnn-configuration)

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

## Change Log

##### v0.3.2

* Change ZipNN suffix from .zpn to .znn 


##### v0.3.1

* Prepare dtype16 (BF16 and FP16) for multi-threading by changing its C logic. For each chunk, byte ordering, bit ordering, and compression are processed separately.

* Integrate the Streaming support into zipnn python code.


##### v0.2.4

* Add support for Streaming when using outside scripts

* Fix bug: Compression didn't work when compressing files larger than 3GB


##### v0.2.3

* Change the byte ordering implementation to C (for better performance).

* Change the bfloat16/float16 implementation to a C implementation with Huffman encoding, running on chunks of 256KB each.
  
* Float 32 using ZSTD compression as in v0.1.1

* Add support with uint32 with ZSTD compression.

##### v0.1.1

* Python implementation of compressing Models, float32, float15, bfloat16 with byte ordering and ZSTD.


## Cite
```
@article{hershcovitch2024lossless,
  title={Lossless and Near-Lossless Compression for Foundation Models},
  author={Hershcovitch, Moshik and Choshen, Leshem and Wood, Andrew and Enmouri, Ilias and Chin, Peter and Sundararaman, Swaminathan and Harnik, Danny},
  journal={arXiv preprint arXiv:2404.15198},
  year={2024}
}
```
