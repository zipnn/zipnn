# ZipNN Scripts

Below is a brief overview of each script available in the `scripts` folder:

### `zipnn_compress.py`

- **Purpose**: Compresses a file using ZipNN and saves it with the `.zpn` extension.
- **Arguments**:
  - **Required**: Input file name or path.
  - **Optional**:
    - `--float32`: Specifies that the data type is `float32`. If not provided, the default is `bfloat16`.
    - `--streaming_chunk_size`: Specifies the chunk size for streaming during compression. The default is 1MB. Accepts either:
      - An integer (e.g., `1024`), interpreted as bytes.
      - A string with a unit suffix (e.g., `4KB`, `2MB`, `1GB`), where the unit is interpreted as kilobytes, megabytes, or gigabytes.


### `zipnn_decompress.py`

- **Purpose**: Decompresses a `.zpn` file created by ZipNN, restoring it to its original format.
- **Arguments**:
  - **Required**: Input file name or path.
  - **Optional**:
    - `--float32`: Indicates that the data type is `float32`. The default is `bfloat16`.

### `run_zipnn_compression.py`

- **Purpose**: Compresses all files with a specified suffix using ZipNN.
- **Arguments**:
  - **Required**: Suffix of the files to compress (e.g., `.bin` to compress all `.bin` files).
  - **Optional**:
    - `--float32`: Specifies that the data type is `float32`. If not provided, the default is `bfloat16`.
    - `--streaming_chunk_size`: Specifies the chunk size for streaming during compression. The default is 1MB. Accepts either:
      - An integer (e.g., `1024`), interpreted as bytes.
      - A string with a unit suffix (e.g., `4KB`, `2MB`, `1GB`), where the unit is interpreted as kilobytes, megabytes, or gigabytes.
    - `--path`: Path to the folder containing all files that need compression. If left empty, it will look for all files in the current folder.

### `run_zipnn_decompression.py`

- **Purpose**: Decompresses all files with a `.zpn` suffix, removing the `.zpn` extension from the output file names.
- **Arguments**: 
  - **Optional**:
    - `--float32`: Specifies that the data type is `float32`. If not provided, the default is `bfloat16`.
    - `--path`: Path to the folder containing all files that need decompression. If left empty, it will look for all files in the current folder.

To use these scripts, simply copy the desired file to your project directory and run it as needed.
