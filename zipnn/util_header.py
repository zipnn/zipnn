# util for ZipNN Header


class EnumMethod:
    ZSTD = 1
    LZ4 = 2
    SNAPPY = 3

    def method_to_enum(method: str) -> int:
        """
        Takes the method compression name input, and returns the respective enum int value.

        Parameters
        -------------------------------------
        method: string
                Compression method.

        Returns
        -------------------------------------
        Enum int value of the chosen compression method.
        """
        if method in ("zstd", "ZSTD"):
            return EnumMethod.ZSTD
        if method in ("lz4", "LZ4"):
            return EnumMethod.LZ4
        if method in ("snappy", "SNAPPY"):
            return EnumMethod.SNAPPY
        raise ValueError("method ZSTD/LZ4/SNAPPY")


class EnumLossy:
    NONE = 0
    INTEGER = 1
    UNSIGN = 2

    def lossy_to_enum(lossy_type: str) -> int:
        """
        Takes the lossy compression type input, and returns the respective enum int value.

        Parameters
        -------------------------------------
        lossy_type: string
                Lossy compression type.

        Returns
        -------------------------------------
        Enum int value of the chosen compression lossy compression type.
        """
        if lossy_type is None:
            return EnumLossy.NONE
        if lossy_type in ("integer", "INTEGER"):
            return EnumLossy.INTEGER
        if lossy_type in ("unsign", "UNSIGN"):
            return EnumLossy.UNSIGN
        raise ValueError("Lossy compression None/integer/unsign")


def bools_to_bitmask(bools) -> bytes:
    """
    Constructs a bitmask by setting bits corresponding to the indices of True values in a list of booleans, then converts the bitmask to bytes.

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
