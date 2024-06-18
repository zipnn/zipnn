import sys
import struct
from enum import Enum
import torch
import numpy as np
from zipnn.util_header import EnumFormat

@torch.jit.script
def zipnn_multiply_if_max_below(tensor: torch.Tensor, max_val: float, multiplier: float, dtype: int):
    """
    TorchScript function to modify tensor if needed for lossy compression.

    Parameters
    -------------------------------------
    tensor: torch.Tensor
            Torch tensor data.

    max_val: float
            Max value variable used to deicde if torch should be modified.

    multiplier: float
            Tensor multiplier for modification.

    dtype: int
            Torch tensor data type.

    Returns
    -------------------------------------
    Tensor data, and a flag value is_int if it was modified or not.
    """
    is_int = False
    if tensor.abs().max().item() < max_val:
        new_tensor = tensor * multiplier
        new_tensor = new_tensor.to(dtype)
        is_int = True
        return new_tensor, is_int
    return tensor, is_int


@torch.jit.script
def zipnn_divide_int(tensor: torch.Tensor, divisor: float):
    """
    TorchScript function to modify tensor back to original, if it was modified in lossy compression.

    Parameters
    -------------------------------------
    tensor: torch.Tensor
            Torch tensor data.

    is_int: bool
            If true, tensor was modified in compression.

    divisor: float
            Tensor multiplier for modification.

    Returns
    -------------------------------------
    Decompressed tensor data.
    """
    tensor = tensor.to(torch.float32)
    tensor /= divisor
    return tensor


def zipnn_get_dtype_bits(dtype):
    """
    Retrieves dtype of tensor.

    Parameters
    -------------------------------------
    dtype: Tensor.dtype
            The dtype of torch.Tensor object.

    Returns
    -------------------------------------
    Number of bits occupied by the type and the tensor type.
    """
    if dtype.is_floating_point:
        bit_size = torch.finfo(dtype).bits
        if bit_size == 32:
            return bit_size, torch.int32
        if bit_size == 16:
            return bit_size, torch.int16
        sys.exit(f"Error: {dtype} is not float 16, 32")
    sys.exit(f"Error: {dtype} is not a floating point type")


def zipnn_pack_shape(shape):
    """
    Packs the dimensions of a tensor into a byte array, using different size indicators based on the magnitude of each dimension.

    Parameters
    -------------------------------------
    shape: Tensor.shape
            The shape of torch.Tensor object.

    Returns
    -------------------------------------
    Byte data of the packed dimensions.
    """
    packed_data = bytearray()
    packed_data.append(len(shape))  # First byte is the number of dimensions

    for dim in shape:
        if dim < 256:
            packed_data.append(1)  # Append size indicator for 1 byte
            packed_data.extend(struct.pack("B", dim))  # Append actual dimension value
        elif dim < 65536:
            packed_data.append(2)  # Append size indicator for 2 bytes
            packed_data.extend(struct.pack("H", dim))  # Append actual dimension value
        elif dim < 4294967296:
            packed_data.append(4)  # Append size indicator for 4 bytes
            packed_data.extend(struct.pack("I", dim))  # Append actual dimension value
        else:
            packed_data.append(8)  # Append size indicator for 8 bytes
            packed_data.extend(struct.pack("Q", dim))  # Append actual dimension value
    return bytes(packed_data)


def zipnn_unpack_shape(packed_data):
    """
    Unpacks the dimensions of a tensor from a byte array

    Parameters
    -------------------------------------
    packed_data: byte
            Bytes object containing the packed dimensions of the tensor,

    Returns
    -------------------------------------
    A tuple containing the unpacked dimensions as a tuple of integers and the total number of bytes read.
    """
    num_dimensions = packed_data[0]  # Get the number of dimensions from the first byte
    dimensions = []
    i = 1  # Index to start reading dimensions
    total_bytes_read = 1  # Start with 1 byte read for the dimension count

    while i < len(packed_data) and len(dimensions) < num_dimensions:
        size_indicator = packed_data[i]
        total_bytes_read += 1
        i += 1
        if size_indicator == 1:
            (dim,) = struct.unpack("B", packed_data[i : i + 1])
            i += 1
            total_bytes_read += 1
        elif size_indicator == 2:
            (dim,) = struct.unpack("H", packed_data[i : i + 2])
            i += 2
            total_bytes_read += 2
        elif size_indicator == 4:
            (dim,) = struct.unpack("I", packed_data[i : i + 4])
            i += 4
            total_bytes_read += 4
        else:
            (dim,) = struct.unpack("Q", packed_data[i : i + 8])
            i += 8
        dimensions.append(dim)
    return tuple(dimensions), total_bytes_read


def zipnn_is_floating_point(data_format_value, data):
    if (data_format_value == EnumFormat.TORCH.value):
        return torch.is_floating_point(data)
    if (data_format_value == EnumFormat.NUMPY.value):
        return np.issubdtype(data.dtype, np.floating)


class ZipNNDtypeEnum(Enum):
    NONE = (None, None, 0)
    FLOAT32 = (torch.float32, np.float32, 1)  # 32 bits
    FLOAT = (torch.float, np.float32, 2)  # 32 bits
    FLOAT64 = (torch.float64, np.float64, 3)  # 64 bits
    FLOAT16 = (torch.float16, np.float16, 4)  # 16 bits
    HALF = (torch.half, np.float16, 5)  # 16 bits
    BFLOAT16 = (torch.bfloat16, None, 6)  # 16 bits (NumPy does not have bfloat16)
    COMPLEX32 = (torch.complex32, None, 7)  # 32 bits (NumPy does not have complex32)
    CHALF = (torch.complex32, None, 8)  # 32 bits (NumPy does not have complex32)
    COMPLEX64 = (torch.complex64, np.complex64, 9)  # 64 bits
    CFLOAT = (torch.cfloat, np.complex64, 10)  # 64 bits
    COMPLEX128 = (torch.complex128, np.complex128, 11)  # 128 bits
    CDOUBLE = (torch.cdouble, np.complex128, 12)  # 128 bits
    UINT8 = (torch.uint8, np.uint8, 13)  # 8 bits unsigned
    UINT16 = (torch.uint16, np.uint16, 14)  # 16 bits unsigned (Limited support in PyTorch)
    UINT32 = (torch.uint32, np.uint32, 15)  # 32 bits unsigned (Limited support in PyTorch)
    UINT64 = (torch.uint64, np.uint64, 16)  # 64 bits unsigned (Limited support in PyTorch)
    INT8 = (torch.int8, np.int8, 17)  # 8 bits signed
    INT16 = (torch.int16, np.int16, 18)  # 16 bits signed
    SHORT = (torch.int16, np.int16, 19)  # 16 bits signed
    INT32 = (torch.int32, np.int32, 20)  # 32 bits signed
    INT = (torch.int32, np.int32, 21)  # 32 bits signed
    INT64 = (torch.int64, np.int64, 22)  # 64 bits signed
    LONG = (torch.int64, np.int64, 23)  # 64 bits signed
    BOOL = (torch.bool, np.bool_, 24)  # 1 bit
    QUINT8 = (torch.quint8, None, 25)  # 8 bits unsigned quantized (NumPy does not have quantized dtypes)
    QINT8 = (torch.qint8, None, 26)  # 8 bits signed quantized
    QINT32 = (torch.qint32, None, 27)  # 32 bits signed quantized
    QUINT4X2 = (torch.quint4x2, None, 28)  # 4 bits unsigned quantized
    FLOAT8_E4M3FN = (torch.float8_e4m3fn, None, 29)  # 8 bits (Limited support in PyTorch)
    FLOAT8_E5M2 = (torch.float8_e5m2, None, 30)  # 8 bits (Limited support in PyTorch)

    def __init__(self, torch_dtype, numpy_dtype, code):
        self.torch_dtype = torch_dtype
        self.numpy_dtype = numpy_dtype
        self.code = code

    @classmethod
    def from_dtype(cls, dtype):
        for member in cls:
            if dtype == member.torch_dtype or dtype == member.numpy_dtype:
                return member
        return cls.NONE
