# util for ZipNN Header
from enum import Enum


class EnumMethod(Enum):
    AUTO = 0
    HUFFMAN = 1
    ZSTD = 2
    LZ4 = 3
    SNAPPY = 4

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            value = value.upper()
            if value in cls.__members__:
                return cls.__members__[value]


class EnumFormat(Enum):
    BYTE = 1
    TORCH = 2
    NUMPY = 3
    FILE = 4  # Note: I changed this from 3 to 4 to avoid duplicate values unless that's intended

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            value = value.upper()
            if value in cls.__members__:
                return cls.__members__[value]


class EnumLossy(Enum):
    NONE = 0
    INTEGER = 1
    UNSIGN = 2

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            value = value.upper()
            if value in cls.__members__:
                return cls.__members__[value]


def bools_to_bitmask(bools) -> bytes:
    """
    Constructs a bitmask by setting bits corresponding to the indices of True values in a list of booleans,
    then converts the bitmask to bytes.

    Parameters
    -------------------------------------
    bools: iterable boolean object
            List of booleans to set the bitmask.

    Returns
    -------------------------------------
    Bitmask byte data.
    """
    bitmask = 0
    for index, value in enumerate(bools):
        if value:  # Check if the boolean value is True
            bitmask |= 1 << index  # Set the bit at 'index' position
    return bytes(bitmask)
