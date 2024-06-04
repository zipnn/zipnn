import sys
import struct
from enum import Enum
import torch


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

class ZipNNTorchDtypeEnum(Enum):
    NONE = (None, 0)
    FLOAT32 = (torch.float32, 1)  # 32 bits
    FLOAT = (torch.float, 2)  # 32 bits (same as FLOAT32)
    FLOAT64 = (torch.float64, 3)  # 64 bits
    DOUBLE = (torch.double, 4)  # 64 bits (same as FLOAT64)
    FLOAT16 = (torch.float16, 5)  # 16 bits
    HALF = (torch.half, 6)  # 16 bits (same as FLOAT16)
    BFLOAT16 = (torch.bfloat16, 7)  # 16 bits
    UINT8 = (torch.uint8, 8)  # 8 bits
    INT8 = (torch.int8, 9)  # 8 bits
    INT16 = (torch.int16, 10)  # 16 bits
    SHORT = (torch.short, 11)  # 16 bits (same as INT16)
    INT32 = (torch.int32, 12)  # 32 bits
    INT = (torch.int, 13)  # 32 bits (same as INT32)
    INT64 = (torch.int64, 14)  # 64 bits
    LONG = (torch.long, 15)  # 64 bits (same as INT64)
    BOOL = (torch.bool, 16)  # 8 bits
    COMPLEX64 = (torch.complex64, 17)  # 64 bits
    CFLOAT = (torch.cfloat, 18)  # 64 bits (same as COMPLEX64)
    COMPLEX128 = (torch.complex128, 19)  # 128 bits
    CDOUBLE = (torch.cdouble, 20)  # 128 bits (same as COMPLEX128)

    def __init__(self, dtype, code):
        self.dtype = dtype
        self.code = code

    @classmethod
    def from_torch_dtype(cls, dtype):
        if dtype is None:
            return cls.NONE
        for member in cls:
            if member.dtype is dtype:
                return member
        raise ValueError(f"No matching TorchDtypeEnum for {dtype}")

    @classmethod
    def from_code(cls, code):
        for member in cls:
            if member.code == code:
                return member
        raise ValueError(f"No matching TorchDtypeEnum for code {code}")

    def __str__(self):
        return f"{self.name} (dtype: {self.dtype}, code: {self.code})"

