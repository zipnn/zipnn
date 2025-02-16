# ZipNN and Hugging Face Integration

## Compress and Upload a Model to Hugging Face (safetensors)
1. Create a destination repository (e.g. myfork) in https://huggingface.co
2. Clone your fork repository: 
```bash
git clone git@hf.co:me/myfork
cd myfork
```
3. Set up Git LFS locally, adds upstream, fetch updates, and pull LFS files.
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

4. Download the scripts for compressing/decompressing AI Models:
```bash
wget -i https://raw.githubusercontent.com/zipnn/zipnn/main/scripts/scripts.txt &&
rm scripts.txt
```

5. Compress the model.
 
```bash
python3 zipnn_compress_path.py safetensors --path .
```

Then, to add the compressed weights to git-lfs tracking and correct the index json:
```bash
git lfs track "*.znn.safetensors" &&
sed -i "" 's/.safetensors/.znn.safetensors/g' model.safetensors.index.json &&
git add *.znn.safetensors .gitattributes model.safetensors.index.json &&
git rm *.safetensors
```

6. Done! Now push the changes as per [the documentation](https://huggingface.co/docs/hub/repositories-getting-started#set-up):
```bash
git lfs install --force --local && # this reinstalls the LFS hooks
huggingface-cli lfs-enable-largefiles . && # needed if some files are bigger than 5GB
git push --force origin main
```

### Using the model
Run our safetensors method before proceeding as normal:

```python
from zipnn import zipnn_safetensors

zipnn_safetensors()

# Load the model from your compressed Hugging Face model card as you normally would
...
```

## Compress and Upload a Model to Hugging Face (other file types)

Steps 1 through 4 are the same.

5. Compress the model.

```bash
python3 zipnn_compress_path.py safetensors --path . --file_compression
```

Then, to add the compressed weights to git-lfs tracking and correct the index json:
```
git lfs track "*.znn" &&
sed -i "" 's/.safetensors/.safetensors.znn/g' model.safetensors.index.json &&
git add *.znn .gitattributes model.safetensors.index.json &&
git rm *.safetensors
```

6. Done! Now push the changes as per [the documentation](https://huggingface.co/docs/hub/repositories-getting-started#set-up):
```bash
git lfs install --force --local && # this reinstalls the LFS hooks
huggingface-cli lfs-enable-largefiles . && # needed if some files are bigger than 5GB
git push --force origin main
```

### Using the model

Run our huggingface method before proceeding as normal:

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

**To run the model, simply add `zipnn_hf()`** at the beginning of the file, and it will take care of decompression for you. By default, the model remains compressed in your local storage, decompressing quickly on the CPU only during loading.


```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from zipnn import zipnn_hf

zipnn_hf()

tokenizer = AutoTokenizer.from_pretrained("royleibov/granite-7b-instruct-ZipNN-Compressed")
model = AutoModelForCausalLM.from_pretrained("royleibov/granite-7b-instruct-ZipNN-Compressed")
```

**Alternatively, you can save the model uncompressed on your local storage.** This way, future loads won’t require a decompression phase.
```
zipnn_hf(replace_local_file=True)
```

**To compress and decompress manually**, simply run:
```bash
python zipnn_compress_path.py safetensors --model royleibov/granite-7b-instruct-ZipNN-Compressed --hf_cache
```

```bash
python zipnn_decompress_path.py --model royleibov/granite-7b-instruct-ZipNN-Compressed --hf_cache
```







You can try other state-of-the-art compressed models from the updating list below:
| ZipNN Compressed Models Hosted on Hugging Face                                                                                      |
|-------------------------------------------------------------------------------------------------------------------------------------|
| [ compressed FacebookAI/roberta-base ]( https://huggingface.co/royleibov/roberta-base-ZipNN-Compressed ) |
| [ compressed meta-llama/Llama-3.2-11B-Vision-Instruct ]( https://huggingface.co/royleibov/Llama-3.2-11B-Vision-Instruct-ZipNN-Compressed ) |
| [compressed ibm-granite/granite-3.0-8b-instruct](https://huggingface.co/royleibov/granite-3.0-8b-instruct-ZipNN-Compressed) |
| [compressed openai/clip-vit-base-patch16](https://huggingface.co/royleibov/clip-vit-base-patch16-ZipNN-Compressed) |
| [compressed jonatasgrosman/wav2vec2-large-xlsr-53-english](https://huggingface.co/royleibov/wav2vec2-large-xlsr-53-english-ZipNN-Compressed) |
| [ compressed mistral-community/pixtral-12b ]( https://huggingface.co/royleibov/pixtral-12b-ZipNN-Compressed ) |
| [ compressed meta-llama/Meta-Llama-3.1-8B-Instruct ]( https://huggingface.co/royleibov/Llama-3.1-8B-ZipNN-Compressed )              |
| [ compressed Qwen/Qwen2-VL-7B-Instruct ]( https://huggingface.co/royleibov/Qwen2-VL-7B-Instruct-ZipNN-Compressed )                  |
| [ compressed ai21labs/Jamba-v0.1 ]( https://huggingface.co/royleibov/Jamba-v0.1-ZipNN-Compressed )                                  |
| [ compressed upstage/solar-pro-preview-instruct ]( https://huggingface.co/royleibov/solar-pro-preview-instruct-ZipNN-Compressed )   |
| [ compressed microsoft/Phi-3.5-mini-instruct ]( https://huggingface.co/royleibov/Phi-3.5-mini-instruct-ZipNN-Compressed )           |
| [compressed ibm-granite/granite-7b-instruct](https://huggingface.co/royleibov/granite-7b-instruct-ZipNN-Compressed) |
| [ compressed ibm-granite/granite-3b-code-base-128k ]( https://huggingface.co/royleibov/granite-3b-code-base-128k-ZipNN-Compressed ) |  


You can also try one of these python notebooks hosted on Kaggle: [granite 3b](https://www.kaggle.com/code/royleibovitz/huggingface-granite-3b-example), [Llama 3.2](https://www.kaggle.com/code/royleibovitz/huggingface-llama-3-2-example), [phi 3.5](https://www.kaggle.com/code/royleibovitz/huggingface-phi-3-5-example).  

[Click here](../examples/README.md) to explore other examples of compressed models hosted on Hugging Face
