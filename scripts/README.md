# ZipNN Scripts for working with files

Below is a brief overview of each script available in the `scripts` folder:

### `zipnn_compress_file.py`

Usage example:
```
python zipnn_compress_file.py model_name
```
- **Purpose**: Compresses a single file, using ZipNN.
- **Arguments**:
  - **Required**: The path of the file to compress.
  - **Optional**:
    - `--dtype`: Flag that specifies the data type, out of `bfloat16`, `float16`, `float32`. Default is `bfloat16`.
    - `--streaming_chunk_size`: Specifies the chunk size for streaming during compression. The default is 1MB. Accepts either:
      - An integer (e.g., `1024`), interpreted as bytes.
      - A string with a unit suffix (e.g., `4KB`, `2MB`, `1GB`), where the unit is interpreted as kilobytes, megabytes, or gigabytes.
    - `--delete`: Flag that specifies to delete the files instead of compressing them.
    - `--force`: Flag that forces overwriting when compressing.
    - `--hf_cache`: A flag that indicates if the file is in the Hugging Face cache.
    - `--method`: Specify the method to use out of HUFFMAN, ZSTD, FSE, AUTO. Default is HUFFMAN.
    - `--verification`: A flag that verifies that a compression is decompressed correctly.
    - `--test`: A flag to not write the compression to a file.
   
### `zipnn_decompress_file.py`

Usage example:
```
python zipnn_decompress_file.py compressed_model_name.znn
```

- **Purpose**: Decompresses the input file, removing the `.znn` extension from the output file name, using ZipNN.
- **Arguments**: 
  - **Required**: The path of the file to decompress.
  - **Optional**:
    - `--delete`: Flag that specifies to delete the files instead of compressing them.
    - `--force`: Flag that forces overwriting when decompressing.
    - `--hf_cache`: A flag that indicates if the file is in the Hugging Face cache.

### `zipnn_compress_path.py`

Usage example:
```
python zipnn_compress_path.py safetensors  --path data/
```

- **Purpose**: Compresses all files with a specified suffix using ZipNN under a path.
- **Arguments**:
  - **Required**: Suffix of the files to compress (e.g., `.bin` to compress all `.bin` files).
  - **Optional**:
    - `--dtype`: Flag that specifies the data type, out of `bfloat16`, `float16`, `float32`. Default is `bfloat16`.
    - `--streaming_chunk_size`: Specifies the chunk size for streaming during compression. The default is 1MB. Accepts either:
      - An integer (e.g., `1024`), interpreted as bytes.
      - A string with a unit suffix (e.g., `4KB`, `2MB`, `1GB`), where the unit is interpreted as kilobytes, megabytes, or gigabytes.
    - `--path`: Path to the folder containing all files that need compression. If left empty, it will look for all files in the current folder.
    - `--delete`: Flag that specifies to delete the files instead of compressing them.
    - `-r`,`--recursive`: Both flags operate the same: they specify to look recursively in all subdirectories (of current folder or of the path given) for files with the specified suffix.
    - `--force`: Flag that forces overwriting when compressing.
    - `--max_processes`: Amount of max processes that can be used during the compression. The default is 1.
    - `--model`: Only when using --hf_cache, specify the model name or path. E.g. 'ibm-granite/granite-7b-instruct'.
    - `--model_branch`: Only when using --model, specify the model branch. Default is 'main'.
    - `--hf_cache`: A flag that indicates if the file is in the Hugging Face cache. Must either specify --model or --path to the model's snapshot cache.
    - `--method`: Specify the method to use out of HUFFMAN, ZSTD, FSE, AUTO. Default is HUFFMAN.
    - `--verification`: A flag that verifies that a compression is decompressed correctly.
    - `--test`: A flag to not write the compression to a file.

### `zipnn_decompress_path.py`

Usage example:
```
python zipnn_decompress_path.py --path data/
```

- **Purpose**: Decompresses all files with a `.znn` suffix under a path, removing the `.znn` extension from the output file name, using ZipNN.
- **Arguments**: 
  - **Optional**:
    - `--path`: Path to the folder containing all files that need decompression. If left empty, it will look for all files in the current folder.
    - `--delete`: Flag that specifies to delete the files instead of compressing them.
    - `--force`: Flag that forces overwriting when decompressing.
    - `--max_processes`: Amount of max processes that can be used during the decompression. The default is 1.
    - `--model`: Only when using --hf_cache, specify the model name or path. E.g. 'ibm-granite/granite-7b-instruct'.
    - `--model_branch`: Only when using --model, specify the model branch. Default is 'main'.
    - `--hf_cache`: A flag that indicates if the file is in the Hugging Face cache. Must either specify --model or --path to the model's snapshot cache.

To use these scripts, simply copy the desired file to your project directory and run it as needed.

### `zipnn_compress_file_delta.py`

Usage example:
```
python zipnn_compress_file_delta.py model_name delta_model_name
```
- **Purpose**: Compresses a single file, using ZipNN's delta compression.
- **Arguments**:
  - **Required**:
    - `--input_file`: Specify the path to the file to compress.
    - `--delta_file`: Specify the path to the delta file.
  - **Optional**:
    - `--dtype`: Flag that specifies the data type, out of `bfloat16`, `float16`, `float32`. Default is `bfloat16`.
    - `--streaming_chunk_size`: Specifies the chunk size for streaming during compression. The default is 1MB. Accepts either:
      - An integer (e.g., `1024`), interpreted as bytes.
      - A string with a unit suffix (e.g., `4KB`, `2MB`, `1GB`), where the unit is interpreted as kilobytes, megabytes, or gigabytes.
    - `--delete`: Flag that specifies to delete the files instead of compressing them.
    - `--force`: Flag that forces overwriting when compressing.
    - `--hf_cache`: A flag that indicates if the file is in the Hugging Face cache.
    - `--method`: Specify the method to use out of HUFFMAN, ZSTD, FSE, AUTO. Default is HUFFMAN.
    - `--verification`: A flag that verifies that a compression is decompressed correctly.
    - `--test`: A flag to not write the compression to a file.
   
### `zipnn_decompress_file_delta.py`

Usage example:
```
python zipnn_decompress_file_delta.py compressed_model_name.znn delta_model_name
```
- **Purpose**: Decompresses a single file, using ZipNN's delta compression, removing the `.znn` extension from the output file name.
- **Arguments**: 
  - **Required**:
    - `--input_file`: Specify the path to the file to decompress.
    - `--delta_file`: Specify the path to the delta file.
  - **Optional**:
    - `--delete`: Flag that specifies to delete the files instead of compressing them.
    - `--force`: Flag that forces overwriting when decompressing.
    - `--hf_cache`: A flag that indicates if the file is in the Hugging Face cache.
