# ZipNN Scripts for working with files

Below is a brief overview of each script available in the `scripts` folder:

### `zipnn_file_compression.py`

Usage example:
```
python zipnn_file_compression.py model_name
```
- **Purpose**: Compresses a single file, using ZipNN.
- **Arguments**:
  - **Required**: The path of the file to compress.
  - **Optional**:
    - `--float32`: Flag that specifies that the data type is `float32`. If not provided, the default is `bfloat16`.
    - `--streaming_chunk_size`: Specifies the chunk size for streaming during compression. The default is 1MB. Accepts either:
      - An integer (e.g., `1024`), interpreted as bytes.
      - A string with a unit suffix (e.g., `4KB`, `2MB`, `1GB`), where the unit is interpreted as kilobytes, megabytes, or gigabytes.
    - `--delete`: Flag that specifies to delete the files instead of compressing them.
    - `--force`: Flag that forces overwriting when compressing.
   
### `zipnn_file_decompression.py`

Usage example:
```
python zipnn_file_decompression.py model_name.zpn
```

- **Purpose**: Decompresses the input file, removing the `.zpn` extension from the output file name, using ZipNN.
- **Arguments**: 
  - **Required**: The path of the file to decompress.
  - **Optional**:
    - `--float32`: Flag that specifies that the data type is `float32`. If not provided, the default is `bfloat16`.
    - `--delete`: Flag that specifies to delete the files instead of compressing them.
    - `--force`: Flag that forces overwriting when decompressing.

### `run_zipnn_compression.py`

Usage example:
```
python run_compression_zipnn.py safetensors  --path data/
```

- **Purpose**: Compresses all files with a specified suffix using ZipNN.
- **Arguments**:
  - **Required**: Suffix of the files to compress (e.g., `.bin` to compress all `.bin` files).
  - **Optional**:
    - `--float32`: Flag that specifies that the data type is `float32`. If not provided, the default is `bfloat16`.
    - `--streaming_chunk_size`: Specifies the chunk size for streaming during compression. The default is 1MB. Accepts either:
      - An integer (e.g., `1024`), interpreted as bytes.
      - A string with a unit suffix (e.g., `4KB`, `2MB`, `1GB`), where the unit is interpreted as kilobytes, megabytes, or gigabytes.
    - `--path`: Path to the folder containing all files that need compression. If left empty, it will look for all files in the current folder.
    - `--delete`: Flag that specifies to delete the files instead of compressing them.
    - `-r`,`--recursive`: Both flags operate the same: they specify to look recursively in all subdirectories (of current folder or of the path given) for files with the specified suffix.
    - `--force`: Flag that forces overwriting when compressing.
    - `--max_processes`: Amount of max processes that can be used during the compression. The default is 1.

### `run_zipnn_decompression.py`

Usage example:
```
python run_decompression_zipnn.py --path data/
```

- **Purpose**: Decompresses all files with a `.zpn` suffix, removing the `.zpn` extension from the output file name, using ZipNN.
- **Arguments**: 
  - **Optional**:
    - `--float32`: Flag that specifies that the data type is `float32`. If not provided, the default is `bfloat16`.
    - `--path`: Path to the folder containing all files that need decompression. If left empty, it will look for all files in the current folder.
    - `--delete`: Flag that specifies to delete the files instead of compressing them.
    - `--force`: Flag that forces overwriting when decompressing.
    - `--max_processes`: Amount of max processes that can be used during the decompression. The default is 1.

To use these scripts, simply copy the desired file to your project directory and run it as needed.
