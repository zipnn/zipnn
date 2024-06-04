# util for ZipNN Header
from enum import Enum


class EnumMethod(Enum):
    ZSTD = 1
    LZ4 = 2
    SNAPPY = 3

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
