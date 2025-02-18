# ZipNN Examples

In this folder some ZipNN usage example files are showcased.

To test an example yourself, simply run
```bash
python3 EXAMPLE_FILE.py
```


## Run vLLM with ZipNN 


### Run vanilla vLLM


Example running GPT2:

```
vllm serve zipnn/gpt2-ZipNN --gpu-memory-utilization 0.9 --max-num-batched-tokens 8192
```

or 

```
python3 -m vllm.entrypoints.openai.api_server --model gpt2 --gpu-memory-utilization 0.9 --max-num-batched-tokens 8192
```
### Run vllm with ZipNN 

Example running Compressed GPT2:

Add ZipNN Safetensors plugin and you can run vLLM with compressed models

```
python3 -c "from zipnn import zipnn_safetensors; zipnn_safetensors(); import runpy; runpy.run_module('vllm.entrypoints.openai.api_server', run_name='__main__', alter_sys=True)" --model zipnn/gpt2-ZipNN --gpu-memory-utilization 0.9 --max-num-batched-tokens 8192
```


## ZipNN Docker Example:

### Run ZipNN vLLM Docker:
[zipnn/vllm-openai Docker on dockerhub](https://hub.docker.com/r/zipnn/vllm-openai)

Run with a docker that already has a ZipNN


Example running GPT2:

```
sudo docker run --runtime=nvidia --gpus all --shm-size 1g -p 8000:8000 zipnn/vllm-openai --model zipnn/gpt2-ZipNN --gpu-memory-utilization 0.9 --max-num-batched-tokens 8192
```

###  Run vanilla vLLM Docker:

Example running Compressed GPT2:

```
sudo docker run --runtime=nvidia --gpus all --shm-size 1g -p 8000:8000 vllm/vllm-openai --model gpt2 --gpu-memory-utilization 0.9 --max-num-batched-tokens 8192
```


## Example Queries:

Example Query GPT2:
```
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "gpt2",
        "prompt": "Once upon a time",
        "max_tokens": 50
    }'
```

Example Query compressed GPT2:
```
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "zipnn/gpt2-ZipNN",
        "prompt": "Once upon a time",
        "max_tokens": 50
    }'
```


# Notebooks on Kaggle: 

There are also usage examples hosed on Kaggle: [granite 3b](https://www.kaggle.com/code/royleibovitz/huggingface-granite-3b-example), [Llama 3.2](https://www.kaggle.com/code/royleibovitz/huggingface-llama-3-2-example), [phi 3.5](https://www.kaggle.com/code/royleibovitz/huggingface-phi-3-5-example).  
