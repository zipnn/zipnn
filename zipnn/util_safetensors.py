"""
Utils for handling safetensors files.
"""
from typing import Dict, TypedDict
import json
import torch


METADATA_KEY = "znn_compressed_vectors"


COMPRESSION_METHOD = "HUFFMAN"
COMPRESSED_DTYPE = torch.uint8


class CompressedTensorInfo(TypedDict):
    """
    Metadata saved for a compress tensor.

    Attributes:
        dtype (str): The dtype of the underlying uncompressed tensor.
        shape (str): The shape of the underlying uncompressed tensor.
    """
    dtype: str
    shape: str


def build_compressed_tensor_info(uncompressed_tensor: torch.tensor) -> CompressedTensorInfo:
    """
    returns metadata to be saved for the respective compressed tensor.
    """
    dtype = str(uncompressed_tensor.dtype)
    if dtype.startswith('torch.'):
        dtype = dtype[len('torch.'):]

    return CompressedTensorInfo(
        dtype=dtype,
        shape=str(list(uncompressed_tensor.shape)))


def set_compressed_tensors_metadata(
        compressed_tensor_infos: Dict[str, CompressedTensorInfo],
        metadata: Dict[str, str]):
    """
    sets file-level metadata on all compressed tensors.
    """
    if metadata:
        metadata[METADATA_KEY] = json.dumps(compressed_tensor_infos)


def get_compressed_tensors_metadata(metadata: Dict[str, str]) -> Dict[str, CompressedTensorInfo]:
    """
    retrieves file-level metadata on all compressed tensors.
    """
    if metadata:
        return json.loads(metadata.get(METADATA_KEY) or {})
    else:
        return {}
