import sys
import struct
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
def zipnn_divide_int(tensor: torch.Tensor, is_int: bool, divisor: float):
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
    if is_int:
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


class ZipnnEnumTorchDtype:
    NONE: 0
    FLOAT32 = 1  # 32 bits
    FLOAT = 2  # 32 bits
    FLOAT64 = 3  # 64 bits
    DOUBLE = 4  # 64 bits
    FLOAT16 = 5  # 16 bits
    HALF = 6  # 16 bits
    BFLOAT16 = 7  # 16 bits
    UINT8 = 8  # 8 bits
    INT8 = 9  # 8 bits
    INT16 = 10  # 16 bits
    SHORT = 11  # 16 bits
    INT32 = 12  # 32 bits
    INT = 13  # 32 bits
    INT64 = 14  # 32 bits
    LONG = 15  # 32 bits
    BOOL = 16  # 32 bits
    COMPLEX64 = 17  # 64 bits
    CFLOAT = 18  # 64 bits
    COMPLEX128 = 19  # 128 bits
    CDOUBLE = 20  # 128 bits

    def dtype_to_enum(dtype):
        """
        Converts Tensor.dtype to ZipnnEnumTorchDtype enum value.

        Parameters
        -------------------------------------
        dtype: Tensor.dtype
                The dtype of the tensor.

        Returns
        -------------------------------------
        The ZipnnEnumTorchDtype enum value corresponding to the Tensor.dtype input.
        """
        if dtype == torch.float32:
            return ZipnnEnumTorchDtype.FLOAT32
        if dtype == torch.float:
            return ZipnnEnumTorchDtype.FLOAT
        if dtype == torch.float64:
            return ZipnnEnumTorchDtype.FLOAT64
        if dtype == torch.double:
            return ZipnnEnumTorchDtype.DOUBLE
        if dtype == torch.float16:
            return ZipnnEnumTorchDtype.FLOAT16
        if dtype == torch.half:
            return ZipnnEnumTorchDtype.HALF
        if dtype == torch.bfloat16:
            return ZipnnEnumTorchDtype.BFLOAT16
        if dtype == torch.uint8:
            return ZipnnEnumTorchDtype.UINT8
        if dtype == torch.int8:
            return ZipnnEnumTorchDtype.INT8
        if dtype == torch.int16:
            return ZipnnEnumTorchDtype.INT16
        if dtype == torch.short:
            return ZipnnEnumTorchDtype.SHORT
        if dtype == torch.int32:
            return ZipnnEnumTorchDtype.INT32
        if dtype == torch.int:
            return ZipnnEnumTorchDtype.INT
        if dtype == torch.int64:
            return ZipnnEnumTorchDtype.INT64
        if dtype == torch.long:
            return ZipnnEnumTorchDtype.LONG
        if dtype == torch.bool:
            return ZipnnEnumTorchDtype.BOOL
        if dtype == torch.complex64:
            return ZipnnEnumTorchDtype.COMPLEX64
        if dtype == torch.cfloat:
            return ZipnnEnumTorchDtype.CFLOAT
        if dtype == torch.complex128:
            return ZipnnEnumTorchDtype.COMPLEX128
        if dtype == torch.cdouble:
            return ZipnnEnumTorchDtype.CDOUBLE

    def enum_to_dtype(enum: int):
        """
        Converts ZipnnEnumTorchDtype enum value to Tensor.dtype.

        Parameters
        -------------------------------------
        enum: int
                The ZipnnEnumTorchDtype enum value.

        Returns
        -------------------------------------
        The Tensor.dtype corresponding to the ZipnnEnumTorchDtype enum value input.
        """
        if enum == ZipnnEnumTorchDtype.FLOAT32:
            return torch.float32
        if enum == ZipnnEnumTorchDtype.FLOAT:
            return torch.float
        if enum == ZipnnEnumTorchDtype.FLOAT64:
            return torch.float64
        if enum == ZipnnEnumTorchDtype.DOUBLE:
            return torch.double
        if enum == ZipnnEnumTorchDtype.FLOAT16:
            return torch.float16
        if enum == ZipnnEnumTorchDtype.HALF:
            return torch.half
        if enum == ZipnnEnumTorchDtype.BFLOAT16:
            return torch.bfloat16
        if enum == ZipnnEnumTorchDtype.UINT8:
            return torch.uint8
        if enum == ZipnnEnumTorchDtype.INT8:
            return torch.int8
        if enum == ZipnnEnumTorchDtype.INT16:
            return torch.int16
        if enum == ZipnnEnumTorchDtype.SHORT:
            return torch.short
        if enum == ZipnnEnumTorchDtype.INT32:
            return torch.int32
        if enum == ZipnnEnumTorchDtype.INT:
            return torch.int
        if enum == ZipnnEnumTorchDtype.INT64:
            return torch.int64
        if enum == ZipnnEnumTorchDtype.LONG:
            return torch.long
        if enum == ZipnnEnumTorchDtype.BOOL:
            return torch.bool
        if enum == ZipnnEnumTorchDtype.COMPLEX64:
            return torch.complex64
        if enum == ZipnnEnumTorchDtype.CFLOAT:
            return torch.cfloat
        if enum == ZipnnEnumTorchDtype.COMPLEX128:
            return torch.complex128
        if enum == ZipnnEnumTorchDtype.CDOUBLE:
            return torch.cdouble
