# ZipNN and Hugging Face Integration

## Compress and Upload a Model to Hugging Face
0. Fork the model's Hugging Face repo (adapted from [the documentation](https://huggingface.co/docs/hub/en/repositories-next-steps#duplicating-with-the-git-history-fork)):
```bash
git lfs install --skip-smudge --local &&
git remote add upstream git@hf.co:ibm-granite/granite-7b-instruct &&
git fetch upstream &&
git lfs fetch --all upstream
```

* If you want to completely override the fork history (which should only have an initial commit), run:
```bash
git reset --hard upstream/main &&
git lfs pull upstream
```
* If you want to rebase instead of overriding, run the following command and resolve any conflicts:
```bash
git rebase upstream/main &&
git lfs pull upstream
```

1. Compress all the model weights
```bash
python3 zipnn_compress_path.py safetensors --path .
```

2. Add the compressed weights to git-lfs tracking and correct the index json
```
git lfs track "*.znn" &&
sed -i 's/.safetensors/.safetensors.znn/g' model.safetensors.index.json &&
git add *.znn .gitattributes model.safetensors.index.json &&
git rm *.safetensors
```

3. Done! Now push the changes as per [the documentation](https://huggingface.co/docs/hub/repositories-getting-started#set-up):
```bash
git lfs install --force --local && # this reinstalls the LFS hooks
huggingface-cli lfs-enable-largefiles . && # needed if some files are bigger than 5GB
git push --force origin main
```

To use the model simply run our ZipNN Hugging Face method before proceeding as normal:
```python
from zipnn import zipnn_hf

zipnn_hf()

# Load the model from your compressed Hugging Face model card as you normally would
...
```

## Download Compressed Models from Hugging Face
In this example we show how to use the [compressed ibm-granite granite-7b-instruct](https://huggingface.co/royleibov/granite-7b-instruct-ZipNN-Compressed) hosted on Hugging Face.

First, make sure you have ZipNN installed:
```bash
pip install zipnn
```

To run the model, simply add `zipnn_hf()` at the beginning of the file, and it will take care of decompression for you:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from zipnn import zipnn_hf

zipnn_hf()

tokenizer = AutoTokenizer.from_pretrained("royleibov/granite-7b-instruct-ZipNN-Compressed")
model = AutoModelForCausalLM.from_pretrained("royleibov/granite-7b-instruct-ZipNN-Compressed")
```
ZipNN also allows you to seamlessly save local disk space in your cache after the model is downloaded.

To compress the cached model, simply run:
```bash
python zipnn_compress_path.py safetensors --model royleibov/granite-7b-instruct-ZipNN-Compressed --hf_cache
```

The model will be decompressed automatically and safely as long as `zipnn_hf()` is added at the top of the file like in the example above.

To decompress manualy, simply run:
```bash
python zipnn_decompress_path.py --model royleibov/granite-7b-instruct-ZipNN-Compressed --hf_cache
```

You can try other compressed models, like [compressed meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/royleibov/Llama-3.1-8B-ZipNN-Compressed), [compressed Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/royleibov/Qwen2-VL-7B-Instruct-ZipNN-Compressed), [compressed ai21labs/Jamba-v0.1](https://huggingface.co/royleibov/Jamba-v0.1-ZipNN-Compressed).  
[Click here to explore other examples of compressed models hosted on Hugging Face](../examples/README.md)