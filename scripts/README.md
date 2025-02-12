# ZipNN Scripts for File Compression and Decompression

This repository provides a set of command-line scripts designed to efficiently compress and decompress files using ZipNN. The scripts support multiple compression and decompression methods:

- **Standard ZipNN compression**
- **Delta compression**, which requires a delta file and compresses the difference between the input and the delta file
- **Safetensors compression**, which allows tensor-by-tensor compression for safetensors files
- **Path-based compression**, enabling batch compression/decompression for all files in a specified directory

Each script provides configurable options for fine-tuning performance and flexibility, as detailed below.

## Available Scripts

### Compression Scripts
1. **`zipnn_compress_file.py`** - Compresses a single file, adding a `.znn` suffix.
   
   ```
   python zipnn_compress_file.py model_path
   ```
3. **`zipnn_compress_file_delta.py`** - Compresses a file using delta compression, requiring a delta reference file.
   ```
   python zipnn_compress_file_delta.py model_path delta_file_path
   ```

5. **`zipnn_compress_path.py`** - Compresses all files with a given suffix in a specified directory (optionally recursive).
   ```
   python zipnn_compress_path.py suffix_of_files
   ```
7. **`zipnn_compress_safetensors.py`** - Compresses safetensors files tensor by tensor, adding a `.znn.safetensors` suffix.
   ```
   python zipnn_compress_path_safetensors.py safetensors_path
   ```

### Decompression Scripts
1. **`zipnn_decompress_file.py`** - Decompresses a single file.
   
   ```
   python zipnn_decompress_file.py compressed_model_path
   ```
3. **`zipnn_decompress_file_delta.py`** - Decompresses a file using delta compression, requiring a delta reference file.
   ```
   python zipnn_decompress_file_delta.py compressed_model_path delta_file_path
   ```
5. **`zipnn_decompress_path.py`** - Decompresses all `.znn`, `.znn.safetensors` files in a specified directory (optionally recursive).
   ```
   python zipnn_decompress_path.py 
   ```
7. **`zipnn_decompress_safetensors.py`** - Decompresses safetensors files efficiently, tensor by tensor.
   ```
   python zipnn_decompress_path_safetensors.py compressed_safetensors_path
   ```


## Scripts Configuration

### `zipnn_compress_file.py`

Usage example:
```
python zipnn_compress_file.py model_name
```
- **Purpose**: Compresses a single file, using ZipNN.
- **Arguments**:
  - **Required**: The path of the file to compress.
  - **Optional**:
    - `--dtype`: The data type of the file to be compressed. The options are "bfloat16", "float16", "float32", and "bfloat16" is the default.
    - `--streaming_chunk_size`: Specifies the chunk size for streaming during compression. The default is 1MB. Accepts either:
      - An integer (e.g., `1024`), interpreted as bytes.
      - A string with a unit suffix (e.g., `4KB`, `2MB`, `1GB`), where the unit is interpreted as kilobytes, megabytes, or gigabytes.
    - `--delete`: Flag that specifies deleting the original file after compression.
    - `--force`: Flag that forces overwriting when compressing.
    - `--hf_cache`: A flag that indicates if the file is in the Hugging Face cache.
    - `--method`: The compression method to be used. The options are "HUFFMAN", "ZSTD", "FSE", "AUTO", and "HUFFMAN" is the default.
    - `--verification`: A flag that verifies that a compression can be decompressed correctly.
    - `--test`: A flag to not write the compressed data to a file.
    - `--is_streaming`: A flag to compress using streaming.

   
### `zipnn_decompress_file.py`

Usage example:
```
python zipnn_decompress_file.py compressed_model_name.znn
```

- **Purpose**: Decompresses the input file, removing the `.znn` extension from the output file name, using ZipNN.
- **Arguments**: 
  - **Required**: The path of the file to decompress.
  - **Optional**:
    - `--delete`: Flag that specifies deleting the compressed file after decompression.
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
    - `--dtype`: The data type of the file to be compressed. The options are "bfloat16", "float16", "float32", and "bfloat16" is the default.
    - `--streaming_chunk_size`: Specifies the chunk size for streaming during compression. The default is 1MB. Accepts either:
      - An integer (e.g., `1024`), interpreted as bytes.
      - A string with a unit suffix (e.g., `4KB`, `2MB`, `1GB`), where the unit is interpreted as kilobytes, megabytes, or gigabytes.
    - `--path`: Path to the folder containing all files that need compression. If left empty, it will look for all files in the current folder.
    - `--delete`: Flag that specifies deleting the original files after compression
    - `-r`,`--recursive`: Both flags operate the same: they specify to look recursively in all subdirectories (of current folder or of the path given) for files with the specified suffix.
    - `--force`: Flag that forces overwriting when compressing.
    - `--max_processes`: Amount of max processes that can be used during the compression. The default is 1.
    - `--model`: Only when using --hf_cache, specify the model name or path. E.g. 'ibm-granite/granite-7b-instruct'.
    - `--model_branch`: Only when using --model, specify the model branch. Default is 'main'.
    - `--hf_cache`: A flag that indicates if the file is in the Hugging Face cache. Must either specify --model or --path to the model's snapshot cache.
    - `--method`: The compression method to be used. The options are "HUFFMAN", "ZSTD", "FSE", "AUTO", and "HUFFMAN" is the default.
    - `--verification`: A flag that verifies that a compression can be decompressed correctly.
    - `--test`: A flag to not write the compressed data to a file.
    - `--is_streaming`: A flag to compress using streaming.

### `zipnn_decompress_path.py`

Usage example:
```
python zipnn_decompress_path.py --path data/
```

- **Purpose**: Decompresses all files with a `.znn` suffix under a path, removing the `.znn` extension from the output file name, using ZipNN.
- **Arguments**: 
  - **Optional**:
    - `--path`: Path to the folder containing all files that need decompression. If left empty, it will look for all files in the current folder.
    - `--delete`: Flag that specifies deleting the compressed files after decompression.
    - `--force`: Flag that forces overwriting when decompressing.
    - `--max_processes`: Amount of max processes that can be used during the decompression. The default is 1.
    - `--model`: Only when using --hf_cache, specify the model name or path. E.g. 'ibm-granite/granite-7b-instruct'.
    - `--model_branch`: Only when using --model, specify the model branch. Default is 'main'.
    - `--hf_cache`: A flag that indicates if the file is in the Hugging Face cache. Must either specify --model or --path to the model's snapshot cache.

To use these scripts, simply copy the desired file to your project directory and run it as needed.

### `zipnn_compress_file_delta.py`

Usage example:
```
python zipnn_compress_file_delta.py input_file delta_file
```
- **Purpose**: Compresses a single file with ZipNN using the delta compression method.
- **Arguments**:
  - **Required**:
    - `input_file`: The path of the file to compress using delta comrpession.
    - `delta_file`: The path of the delta file.
  - **Optional**:
    - `--dtype`: The data type of the file to be compressed. The options are "bfloat16", "float16", "float32", and "bfloat16" is the default.
    - `--streaming_chunk_size`: Specifies the chunk size for streaming during compression. The default is 1MB. Accepts either:
      - An integer (e.g., `1024`), interpreted as bytes.
      - A string with a unit suffix (e.g., `4KB`, `2MB`, `1GB`), where the unit is interpreted as kilobytes, megabytes, or gigabytes.
    - `--delete`: Flag that specifies deleting the original file after compression.
    - `--force`: Flag that forces overwriting when compressing.
    - `--hf_cache`: A flag that indicates if the file is in the Hugging Face cache.
    - `--method`: The compression method to be used. The options are "HUFFMAN", "ZSTD", "FSE", "AUTO", and "HUFFMAN" is the default.
    - `--verification`: A flag that verifies that a compression can be decompressed correctly.
    - `--test`: A flag to not write the compressed data to a file.
    - `--is_streaming`: A flag to compress using streaming.

   
### `zipnn_decompress_file_delta.py`

Usage example:
```
python zipnn_decompress_file_delta.py compressed_model_name.znn delta_file
```

- **Purpose**: Decompresses a single file with ZipNN using the delta decompression method.
- **Arguments**: 
  - **Required**:
    - `input_file`: The path of the file to decompress using delta comrpession.
    - `delta_file`: The path of the delta file.
  - **Optional**:
    - `--delete`: Flag that specifies deleting the compressed file after decompression.
    - `--force`: Flag that forces overwriting when decompressing.
    - `--hf_cache`: A flag that indicates if the file is in the Hugging Face cache.


**Examples of compressing Hugging Face models with ZipNN scripts:**

Use --model to specify the full model name from Hugging Face, and --hf_cache for caching:

```bash
python zipnn_compress_path.py safetensors --model royleibov/granite-7b-instruct-ZipNN-Compressed --hf_cache
```

```bash
python zipnn_decompress_path.py --model royleibov/granite-7b-instruct-ZipNN-Compressed --hf_cache
```

