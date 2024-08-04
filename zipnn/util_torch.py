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


def zipnn_is_floating_point(data_format_value, data, bytearray_dtype):
    if data_format_value == EnumFormat.TORCH.value:
        return torch.is_floating_point(data)
    if data_format_value == EnumFormat.NUMPY.value:
        return np.issubdtype(data.dtype, np.floating)
    if data_format_value == EnumFormat.BYTE.value:
        return bytearray_dtype in ("float64", "float32", "float16", "bfloat16")


from enum import Enum
import torch
import numpy as np


class ZipNNDtypeEnum(Enum):
    NONE = (None, None, None, "none", 0)
    FLOAT32 = (torch.float32, np.float32, float, "float32", 1)
    FLOAT = (torch.float, np.float32, float, "float", 2)
    FLOAT64 = (torch.float64, np.float64, float, "float64", 3)
    FLOAT16 = (torch.float16, np.float16, None, "float16", 4)
    HALF = (torch.half, np.float16, None, "half", 5)
    BFLOAT16 = (torch.bfloat16, None, None, "bfloat16", 6)
    COMPLEX32 = (torch.complex32, None, None, "complex32", 7)
    CHALF = (torch.complex32, None, None, "chalf", 8)
    COMPLEX64 = (torch.complex64, np.complex64, complex, "complex64", 9)
    CFLOAT = (torch.cfloat, np.complex64, complex, "cfloat", 10)
    COMPLEX128 = (torch.complex128, np.complex128, complex, "complex128", 11)
    CDOUBLE = (torch.cdouble, np.complex128, complex, "cdouble", 12)
    UINT8 = (torch.uint8, np.uint8, None, "uint8", 13)
    # Torch has limited support (omit it at this stage)
    UINT16 = (None, np.uint16, None, "uint16", 14)
    # Torch has limited support (omit it at this stage)
    UINT32 = (None, np.uint32, None, "uint32", 15)
    # Torch has limited support (omit it at this stage)
    UINT64 = (None, np.uint64, None, "uint64", 16)
    INT8 = (torch.int8, np.int8, None, "int8", 17)
    INT16 = (torch.int16, np.int16, None, "int16", 18)
    SHORT = (torch.int16, np.int16, None, "short", 19)
    INT32 = (torch.int32, np.int32, int, "int32", 20)
    INT = (torch.int32, np.int32, int, "int", 21)
    INT64 = (torch.int64, np.int64, int, "int64", 22)
    LONG = (torch.int64, np.int64, int, "long", 23)
    BOOL = (torch.bool, np.bool_, bool, "bool", 24)
    QUINT8 = (torch.quint8, None, None, "quint8", 25)
    QINT8 = (torch.qint8, None, None, "qint8", 26)
    QINT32 = (torch.qint32, None, None, "qint32", 27)
    QUINT4X2 = (torch.quint4x2, None, None, "quint4x2", 28)
    FLOAT8_E4M3FN = (torch.float8_e4m3fn, None, None, "float8_e4m3fn", 29)
    FLOAT8_E5M2 = (torch.float8_e5m2, None, None, "float8_e5m2", 30)

    def __init__(self, torch_dtype, numpy_dtype, python_dtype, dtype_str, code):
        self.torch_dtype = torch_dtype
        self.numpy_dtype = numpy_dtype
        self.python_dtype = python_dtype
        self.dtype_str = dtype_str
        self.code = code

    @classmethod
    def from_dtype(cls, dtype):
        if isinstance(dtype, str):
            dtype = dtype.lower()
        for member in cls:
            if dtype == member.torch_dtype or dtype == member.numpy_dtype or dtype == member.python_dtype or dtype == member.dtype_str:
                return member
        return cls.NONE
