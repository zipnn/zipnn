# to build, run:
# docker build --build-arg BASE_IMAGE=<base_image:tag> -t <target_image:tag> .
# for example:
# docker build --build-arg BASE_IMAGE=vllm/vllm-openai:latest -t vllm-openai:zipnn .
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

USER root
RUN pip install zipnn

# add a usercustomize script that runs zipnn_safetensors
RUN python3 -c "from site import getsitepackages; import os; path = os.path.join(getsitepackages()[0], 'usercustomize.py'); open(path, 'w').write('from zipnn import zipnn_safetensors\nzipnn_safetensors()\n')"
