# ZipNN Scripts for working with files

Below is a brief overview of each script available in the `scripts` folder:

### `run_zipnn_compression.py`

- **Purpose**: Compresses all files with a specified suffix, or a single file, using ZipNN.
- **Arguments**:
  - **Required**: Suffix of the files to compress (e.g., `.bin` to compress all `.bin` files). If a single file name is provided, only that file will be compressed.
  - **Optional**:
    - `--float32`: Flag that specifies that the data type is `float32`. If not provided, the default is `bfloat16`.
    - `--streaming_chunk_size`: Specifies the chunk size for streaming during compression. The default is 1MB. Accepts either:
      - An integer (e.g., `1024`), interpreted as bytes.
      - A string with a unit suffix (e.g., `4KB`, `2MB`, `1GB`), where the unit is interpreted as kilobytes, megabytes, or gigabytes.
    - `--path`: Path to the folder containing all files that need compression. If left empty, it will look for all files in the current folder.
    - `--delete`: Flag that specifies to delete the files instead of compressing them.
    - `--r`: Flag that specifies to look recursively in all subdirectories (of current folder or of the path given) for files with the specified suffix.

### `run_zipnn_decompression.py`

- **Purpose**: Decompresses all files with a `.zpn` suffix, or a single file, removing the `.zpn` extension from the output file name.
- **Arguments**: 
  - **Optional**:
    - `--float32`: Flag that specifies that the data type is `float32`. If not provided, the default is `bfloat16`.
    - `--path`: Path to the folder containing all files that need decompression. If left empty, it will look for all files in the current folder.
    - `--input_file`: Name of file if only a single file needs decompression.
    - `--delete`: Flag that specifies to delete the files instead of compressing them.

To use these scripts, simply copy the desired file to your project directory and run it as needed.
