# ZipNN Scripts

Below is a brief overview of each script available in the `scripts` folder:

### `zipnn_compress.py`

- **Purpose**: Compresses a file using ZipNN and saves it with the `.zpn` extension.
- **Arguments**:
  - **Required**: Input file.
  - **Optional**:
    - `--float32`: Specifies that the data type is `float32`. If not provided, the default is `bfloat16`.
    - `--chunk_size`: Defines the chunk size multiplier (in MB) for compression. The default is 1MB. For example, entering `2` will set the chunk size to 2MB.

### `zipnn_decompress.py`

- **Purpose**: Decompresses a `.zpn` file created by ZipNN, restoring it to its original format.
- **Arguments**:
  - **Required**: Input file.
  - **Optional**:
    - `--float32`: Indicates that the data type is `float32`. The default is `bfloat16`.

### `run_zipnn_compression.py`

- **Purpose**: Compresses all files with a specified suffix using ZipNN.
- **Arguments**:
  - **Required**: Suffix of the files to compress (e.g., `.bin` to compress all `.bin` files).
  - **Optional**:
    - `--float32`: Specifies that the data type is `float32`. If not provided, the default is `bfloat16`.
    - `--chunk_size`: Defines the chunk size multiplier (in MB) for compression. The default is 1MB. For example, entering `2` will set the chunk size to 2MB.

### `run_zipnn_decompression.py`

- **Purpose**: Decompresses all files with a `.zpn` suffix, removing the `.zpn` extension from the output file names.
- **Optional**:
  - `--float32`: Specifies that the data type is `float32`. If not provided, the default is `bfloat16`.

To use these scripts, simply copy the desired file to your project directory and run it as needed.
