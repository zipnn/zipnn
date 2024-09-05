import time
import os
import math
import numpy as np
import torch
import zstandard as zstd
from zipnn.util_header import EnumMethod, EnumFormat, EnumLossy
import split_dtype
from zipnn.util_torch import (
    ZipNNDtypeEnum,
    zipnn_multiply_if_max_below,
    zipnn_get_dtype_bits,
    zipnn_divide_int,
    zipnn_pack_shape,
    zipnn_unpack_shape,
    zipnn_is_floating_point,
)


class ZipNN:

    def __init__(
        self,
        method: str = "auto",
        input_format: str = "byte",
        bytearray_dtype: str = "bfloat16",
        is_monotonic: int = 0,
        threads: int = 1,
        compression_threshold=0.95,
        check_th_after_percent=10,
        byte_reorder: int = 0,
        reorder_signbit: int = 0,
        delta_compressed_type: str = 0,
        lossy_compressed_type: str = 0,
        lossy_compressed_factor=27,
        compression_chunk=256 * 1024,
        is_streaming: bool = False,
        streaming_chunk_kb: int = 1024 * 1024,
        input_file: str = None,
        compressed_file: str = None,
        decompressed_file: str = None,
        zstd_level: int = 3,
        lz4_compression_level: int = 0,
    ):
        """
         Zipnn class is used to compress and decompress data in byte, file, and Torch tensor formats.
         Additionally, there is support for byte grouping, lossy compression, and delta compression,
             allowing you to use some, all, or none of these techniques.

         Parameters
         -------------------------------------
         method: string
                 Chosen compression method. The options are: ‘zstd’/’ZSTD’, 'huffman'/'HUFFMAN' ‘lz4’/’LZ4’, ‘snappy’/’SNAPPY’.
                 Default is ‘AUTO’, which choose the best compression method automatically.

         input_format: string
                 The type of the input, the same will be for the output.
                 The options are ‘byte’, ‘torch’, ‘numpy’. -> ‘file’ is not implemented yet
                 Default is ‘byte’.


         bytearray_dtype: string,
                 Chosen dtype for bytearray: The options are: ‘float32’, ‘uint32’, ‘uint16', 'bfloat16', 'float16'
                 Default is ‘float32’.

         is_monotonic : bool,
                 The  dataset is monotonic.
                 Default is ‘False’.

         threads: int
                 The maximum threads for th ecompression and the byte/bit reorder.
                 If 0, the code decide according to the dataset len.
                 Default is 1

         compression_threshold: float
                 Save original buffer if not compress above the threshold (default value = 0.95).
                 Only relevant for a compression that uses byte grouping.
                 Default is 0.95.

         check_th_after_percent: int
                 Check the compression threshhold after % from the number of chunk and stop compressing if not pass the compression_threshold.
                 Only relevant for a compression that uses byte grouping.
                 Default is 10[%]

         byte_reorder: int
                 Number of grouping.
                 4 Groups - Nu,ber for the group and zero for truncate.
                 [7] - Gorup 0/1 - 4'th Byte
                 [6-5] - Group 0/1/2 - 3'th Byte
                 [4-3] - Group 0/1/2/3 - 2'th Byte
                 [2-0] - Group 0/1/2/3/4 - 1'th Byte
                 for example:
                 bg16: two groups - 0_00_01_010 - decimal 10
                 fp32: four groups - 1_10_11_100 - decimal 220
                 int32: truncate two MSBs - 0_00_01_001 - decimal 9

         reorder_signbit: int
                 This reorder the bits of the float32 or bfloat16 to better compression.
                 If set to zero [defualt], auto decision according to the dtype
                 If set to 255 - no reorder_signbit.
                 If 16,32 - reorder_signbit for bfloat16 or float32 respectively
                 Defualt is 0 [Auto decision]

        delta_compressed_type: string
               NOT IMPLEMENTED YET.
               Type for delta compression.
               Options are 'byte', 'file'.
               Default is "0" (NOT IMPLEMENTED YET).

         lossy_compressed_type: string
                 Type for lossy compression.
                 Supporting only 'integer' ('unsigned' in the future).
                 Only relevant if compression is lossy.
                 Default is "0" (NOT IMPLEMENTED YET).

         lossy_compressed_factor: int
                 Compression factor for lossy compression.
                 Only relevant if compression is lossy.
                 Default is 27.

         compression_chunk: int
                 Chunk size for compression.
                 Cefault is 256KB

         is_streaming: bool
                 NOT IMPLEMENTED YET.
                 If true – signals compression is for a stream of data.
                 Deafult is False.

         streaming_chunk_kb: int
                 Chunk size for streaming.
                 Only relevant if is_steaming is True.
                 Default is 1MB.

         input_file: string
                 Path to the input file.
                 If ‘file’ is the input type – enter file name.
                 Default is ‘byte’.


         compressed_file: string
                 Path to the compressed file.
                 Only relevant if compressed_ret_type is ‘file’.
                 Default is None.

         decompressed_file: string
                 Path to the decompressed file.
                 Only relevant if compressed_ret_type is ‘file’.
                 Defaul is None.

         zstd_level: int
                 Compression level for ‘zstd’ compression.
                 Only relevant if method is ‘zstd’.
                 Default is 3.``

         lz4_compression_level: int
                 Compression level for ‘lz4’ compression.
                 Only relevant if method is ‘lz4’.
                 Deafult is 0.

         Returns
         -------------------------------------
         ZipNN class instance supporting a specific compression and decompression based on the input given.
        """

        self.method = EnumMethod(method).value
        self.input_format = EnumFormat(input_format).value
        self.bytearray_dtype = bytearray_dtype
        self.is_monotonic = is_monotonic

        self.threads = threads
        self.compression_threshold = compression_threshold
        self.check_th_after_percent = check_th_after_percent
        self.byte_reorder = byte_reorder
        self.reorder_signbit = reorder_signbit

        self.delta_compressed_type = delta_compressed_type
        self.lossy_compressed_type = EnumLossy.NONE if lossy_compressed_type is None else EnumLossy(lossy_compressed_type)
        self.lossy_compressed_factor = lossy_compressed_factor

        if (compression_chunk & (compression_chunk - 1)) == 0:
            self.compression_chunk = compression_chunk
        else:
            raise ValueError("compression_chunk must be a number that is a power of 2.")

        if self.input_format!=EnumFormat.BYTE.value and is_streaming:
            raise ValueError("Streaming is currently implemented only for bytes data type.")
        else:
            self.is_streaming=is_streaming
        
        if (streaming_chunk_kb & (streaming_chunk_kb - 1)) == 0:
            self.streaming_chunk_kb = streaming_chunk_kb
        else:
            raise ValueError("streaming_chunk_kb must be a number that is a power of 2.")
        
        self.streaming_chunk_kb = streaming_chunk_kb

        self.input_file = input_file
        self.compressed_file = compressed_file
        self.decompressed_file = decompressed_file

        self.lz4_compression_level = lz4_compression_level

        self._version_major = 0
        self._version_minor = 3
        self._version_tiny = 2
        self._import_dependencies(zstd_level)

        self.header_length = 32
        self._header = bytearray(self.header_length)
        self._ext_header = b""
        self._shape_size = 0
        self._update_header()

    def _import_dependencies(self, zstd_level):
        """
        Importing needed dependencies, based on the ZipNN compression method.

        Parameters
        -------------------------------------
        torch_dtype: string
                If torch_dtype isn't None, then torch needs to be imported for the decompression.
         zstd_level: int
                Compression level for ‘zstd’ compression.

        threads: int
                Number of threads to be used for ‘zstd’ compression.

        Returns
        -------------------------------------
        None.
        """
        if self.method == EnumMethod.ZSTD.value or self.method == EnumMethod.AUTO.value:
            self._zstd_compress = zstd.ZstdCompressor(level=zstd_level, threads=self.threads)
            self._zstd_decompress = zstd.ZstdDecompressor()

        elif self.method == EnumMethod.LZ4.value:
            try:
                global lz4
                import lz4.frame
            except ImportError as exc:
                raise ImportError("LZ4 library is not installed. Please install it to use LZ4 compression.") from exc

        elif self.method == EnumMethod.SNAPPY.value:
            try:
                global snappy
                import snappy
            except ImportError as exc:
                raise ImportError("Snappy library is not installed. Please install it to use Snappy compression.") from exc

        else:
            raise ValueError(f"Unsupported method {self.method}")

        if self.lossy_compressed_type != EnumLossy.NONE:
            if self.input_format != EnumFormat.TORCH.value:
                raise ValueError("When use lossy compression the input have to be torch.tensor")

    def use_var(self, data, class_var):
        """
        Used to update ZipNN attributes. Updates to data if it isn't null, or to the ZipNN class default if it is.

        Parameters
        -------------------------------------
        data:
                data to update some ZipNN attribute.

        Returns
        -------------------------------------
        data if not None, or the class default value.
        """
        if data is not None:
            return data
        return class_var

    # Header: at least 8 Bytes
    # [0:1] 2 Bytes [ZN]
    # [2:4] 3 Bytes [Versions]
    # [5] 1 Byte [byte_reorder]
    # [6] 1 Byte [bit_reorder]
    # [9] 1 Byte [delta compression]
    # [8] 1 Byte [method]
    # [9] 1 Byte [format]
    # [10] 1 Byte [lossy_compress_type]
    # [11] 1 Byte [lossy_compress_factor]
    # [12] 1 Byte [lossy_is_int]
    # [13] 1 Byte [Compression Chunk]
    # [14] 1 Byte [is_streaming, streaming_chunk_kb]
    # [15] = self.dtype
    # [16-23] = original size
    # [24-32] = compressed file size
    # In case the input_format is TORCH or NUMPY we add to self._ext_header the shape of the data in zipnn_pack format

    # byte order for 64bit
    # Not implemented yet

    # Only in case of torch/ numpy
    # torch.shape/ numpy.shape

    def _update_header_lossy(self, lossy_type, lossy_factor, lossy_is_int):
        """
        Updates header with values of lossy compression.
        """
        self._header[10] = lossy_type.value
        self._header[11] = lossy_factor
        self._header[12] = lossy_is_int

    def _update_header_original_len(self, original_len):
        original_bytes_len = (original_len).to_bytes(8, byteorder="little")
        self._header[16:24] = original_bytes_len

    def _update_header_comp_len(self, comp_len):
        """
        Updates header with the overall compression size
        """
        comp_bytes_len = (comp_len + 32).to_bytes(8, byteorder="little")
        self._header[24:32] = comp_bytes_len

    def _update_header_dtype(self, byte_reorder: int, bit_reorder: int, dtype_code: int):
        """
        Updates header with byte_reorder, bit_reorder, dtype_value
        """

        self._header[5] = byte_reorder
        self._header[6] = bit_reorder
        self._header[15] = dtype_code

    def _update_data_shape(self, shape):
        """
        Updates the shpae of the data add to the and of the header
        """
        self._ext_header = zipnn_pack_shape(shape)

    #
    #        Parameters
    #        -------------------------------------
    #        : string
    #                torch_dtype or numpy_dtype or bytearray_dtype
    #                Default is None.
    #
    #        Returns
    #        -------------------------------------
    #        None.

    def _update_header(self, lossy_compressed_type=None, lossy_compressed_factor=None):
        """
        Updates header with dtype if it's torch decompression.

        Parameters
        -------------------------------------
        lossy_compressed_type: string
                ZipNN attribute lossy_compressed_type.
                Default is None.

        lossy_compressed_factor: int
                ZipNN attribute lossy_compressed_factor.
                Default is None.

        Returns
        -------------------------------------
        None.
        """
        self._header[0:2] = "ZN".encode("ascii")  # header ZN
        self._header[2] = self._version_major
        self._header[3] = self._version_minor
        self._header[4] = self._version_tiny
        #        self._header[5] = byte_reorder
        #        self._header[6] = bit_reorder
        self._header[7] = self.method
        self._header[8] = self.input_format
        self._header[9] = 0 if self.delta_compressed_type is None else self.delta_compressed_type
        #        self._header[10] = self.lossy_compressed_type
        #        self._header[11] = self.lossy_compressed_factor
        #        self._header[12] = self._lossy_is_int
        if self.is_streaming:  # MSB is streaming, unsigned & is stremaing
            self._header[13] = 128 + int(math.log(self.streaming_chunk_kb, 2))
        else:
            self._header[13] = 0
        self._header[14] = int(math.log(self.compression_chunk, 2))
        #        self._header[15] = dtype

    def _retrieve_header(self, ba_compress):
        """
        Retrieves header values, and returns header length.

        Parameters
        -------------------------------------
        ba_compress: byte
                Header data compressed to byte array.

        Returns
        -------------------------------------
        The header length.
        """
        mv = memoryview(ba_compress)
        header = mv[:self.header_length]
        if header[0:2].tobytes().decode("ascii") != "ZN":
            raise ValueError("Header should start with ZN")
        self.version_major = int(header[2])
        self.version_minor = int(header[3])
        self.version_tiny = int(header[4])
        self._byte_reorder = int(header[5])
        self._bit_reorder = int(header[6])
        self.method = int(header[7])
        self.input_format = int(header[8])
        self.delta_compressed_type = int(header[9])
        self.lossy_compressed_type = int(header[10])
        self.lossy_compressed_factor = int(header[11])
        self._lossy_is_int = int(header[12])
        streaming_vals = int(header[13])
        if streaming_vals > 127:
            self.is_streaming = 1
            #self.streaming_chunk_kb = 2 ** (128 - streaming_vals)
        else:
            self.is_streaming = 0
        self.compression_chunk = 2 ** header[14]
        self.dtype = int(header[15])
        self.original_len = int.from_bytes(header[16:24], byteorder="little")

        if self.input_format in (EnumFormat.TORCH.value, EnumFormat.NUMPY.value):
            self.shape_bytes, self._shape_size = zipnn_unpack_shape(mv[self.header_length:])
        return self.header_length + self._shape_size

    #################
    ## compression ##
    #################

    def compress(
        self, data, compress_cpu_gpu="cpu", delta_second_data=None, lossy_compressed_type: str = None, lossy_compressed_factor: int = None
    ):  # the data/ delta_second_data is "byte" or "torch" or "file"
        """
        Compress is the ZipNN function used for compression after the configuration is set.

        Parameters
        -------------------------------------
        data: string
                The data to compress. It’s type can be one of the following options: ‘byte’, ‘torch’, ‘file’. If file, enter filename.
                Default is None.

        delta_second_data: string
                If compression is delta compression,then second data is needed. It's type options are ‘byte’, ‘torch’, ‘file’.
                If file, enter filename.
                Default is None.

        compress_cpu_gpu: string
                Compression will be done by choice, in the CPU or GPU.
                Default is cpu.

        lossy_compressed_type: string
                Lossy compression data type, options are ‘byte’, ‘torch’, ‘file’.
                Default is None.

        lossy_compressed_factor: int
                Lossy compression factor.
                Default is None.

        Returns
        -------------------------------------
        Returns the output of one of the following: compress_delta, compress_bin, compress_torch, compress_file
        (depends on the type of the data compressed), which will be the compressed file,
        in the format chosen in the ZipNN class instance configuration.
        """
        if len(data) % 2 != 0:
            data += b'\x00'
        if self.is_streaming and self.input_format == EnumFormat.BYTE.value:
            mv_data = memoryview(data)
            CHUNK_SIZE = self.streaming_chunk_kb
            # Compression into bytearray
            compressed_buffer = bytearray()
            remaining_bytes = len(data)
            offset = 0

            while remaining_bytes > 0:
                chunk_size = min(CHUNK_SIZE, remaining_bytes)
                chunk = mv_data[offset : offset + chunk_size]
                compressed_chunk = self.compress_torch_numpy_byte(chunk, lossy_compressed_type, lossy_compressed_factor)
                if compressed_chunk:
                    compressed_buffer.extend(compressed_chunk)
                offset += chunk_size
                remaining_bytes -= chunk_size
            return compressed_buffer
        else:
            #        if self.delta_compressed_type is not None:
            #            return self.compress_delta(data, delta_second_data, lossy_compressed_type, lossy_compressed_factor)
            return self.compress_torch_numpy_byte(data, lossy_compressed_type, lossy_compressed_factor)

    def compress_method(self, data: bytes):
        """
        Chooses compression based on compression method.

        Parameters
        -------------------------------------
        data: byte
                Data to compress.

        Returns
        -------------------------------------
        Compression of the data in the chosen method.
        """
        if self.method == EnumMethod.ZSTD.value or self.method == EnumMethod.AUTO.value:
            return self._zstd_compress.compress(data)

        if self.method == EnumMethod.LZ4.value:
            return lz4.frame.compress(data)

        if self.method == EnumMethod.SNAPPY.value:
            return snappy.compress(data)
        raise ValueError(f"Unsupported method {self.method}")

    def compress_bin(
        self, ba: bytes, bit_reorder: int, byte_reorder: int, is_review: int, is_float: int, dtype_size: int, shape, skip_split: bool
    ):
        """
        Compresses byte data.

        Parameters
        -------------------------------------
        ba: byte
                Byte data to compress.

        Returns
        -------------------------------------
        Returns a byte array of the header, data, and some metadata.
        """
        compress_bin_time = time.time()
        is_print = 0

        if (self.byte_reorder == 0b1_01_01_001 and dtype_size == 32) or (self.byte_reorder == 0b0_00_01_001 and dtype_size == 16):
            # one group
            stime = time.time()
            ba_comp = self._header + self.compress_method(ba)
            if self.input_format == EnumFormat.BYTE.value:
                self._update_header_comp_len(len(ba_comp))
                return b"".join([self._header] + [ba_comp])
        else:
            stime = time.time()
            bg_ret = []
            bg_len = []
            buf_is_comp = []

            if dtype_size == 32:
                start_time = time.time()

                if skip_split == 0:
                    buf1, buf2, buf3, buf4 = split_dtype.split_dtype32(ba, bit_reorder, byte_reorder, is_review, self.threads)
                else:
                    buf1 = ba
                    buf2 = None
                    buf3 = None
                    buf4 = None

                if is_print:
                    print("reorder ", time.time() - start_time)
                stime = time.time()
                for b in buf1, buf2, buf3, buf4:
                    if b is None:
                        continue
                    start_time = time.time()
                    bg_comp = self.compress_method(b)

                    if len(bg_comp) / (len(b)) < self.compression_threshold:
                        # Save this byte group compressed
                        buf_is_comp.append((1).to_bytes(1, byteorder="little"))
                        bg_len.append(len(bg_comp).to_bytes(length=8, byteorder="little"))
                        bg_ret.append(bg_comp)
                        if is_print:
                            print(f"We compress this byte: {len(bg_comp)/len(b)} time {time.time()-stime}")
                    else:
                        if is_print:
                            print(f"We did't compress this byte: {len(bg_comp)/len(b)} time {time.time()-stime}")
                        # Save the byte group not compressed
                        buf_is_comp.append((0).to_bytes(1, byteorder="little"))
                        bg_len.append(len(b).to_bytes(length=8, byteorder="little"))
                        bg_ret.append(b)
            if self.input_format in (EnumFormat.TORCH.value, EnumFormat.NUMPY.value):
                shape_bytes = zipnn_pack_shape(shape)
                #
                total_length = len(shape_bytes)
                total_length += sum(len(buf) for buf in buf_is_comp)
                total_length += sum(len(bg) for bg in bg_len)
                total_length += sum(len(ret) for ret in bg_ret)
                self._update_header_comp_len(total_length)
                #
                ba_comp = b"".join([self._header] + [shape_bytes] + buf_is_comp + bg_len + bg_ret)
            else:
                #
                total_length = sum(len(buf) for buf in buf_is_comp)
                total_length += sum(len(bg) for bg in bg_len)
                total_length += sum(len(ret) for ret in bg_ret)
                self._update_header_comp_len(total_length)
                #
                ba_comp = b"".join([self._header] + buf_is_comp + bg_len + bg_ret)

            if dtype_size == 16:
                if is_print:
                    start_time = time.time()
                self._update_header_original_len(len(ba))
                if self.input_format in (EnumFormat.TORCH.value, EnumFormat.NUMPY.value):
                    self._update_data_shape(shape)
                python_header = self._header + self._ext_header
                ba_comp = split_dtype.split_dtype16(
                    python_header,
                    ba,
                    bit_reorder,
                    byte_reorder,
                    is_review,
                    self.compression_chunk,
                    self.compression_threshold,
                    self.check_th_after_percent,
                    self.threads,
                )
                if is_print:
                    print("aggregate output bin ", time.time() - start_time)
        if is_print:
            print("total compression ", time.time() - stime)
            print(f"len ba-comp {len(ba_comp)}")
            print(f"len ba {len(ba)}")
            print("compress_bin_time ", time.time() - compress_bin_time)
        return ba_comp

    #    def prepare_file(self, filename: str):
    #        """
    #        Prepare file.
    #
    #        Parameters
    #        -------------------------------------
    #        filename: string
    #                File name to compress.
    #
    #        Returns
    #        -------------------------------------
    #        Byte array of compressed data.
    #        """
    #
    #        raise ImportError("Not implemented Yet")

    #        if not os.path.exists(filename):
    #         raise FileNotFoundError(f"The file at {filename} was not found.")
    #     with open(filename, "rb") as in_file_handler:
    #         data = in_file_handler.read()
    #     ba = self.compress_bin(data)
    #     return ba
    #
    #     # streaming
    #       if (self.is_streaming): # streaming only for input_format == "file_streaming" and compress_ret_type == "file_streaming"
    #          assert (self.compressed_ret_type == "file")
    #          with open(input_path, 'rb') as infile, open(output_path, 'wb') as outfile:
    #              while chunk := infile.read(CHUNK_SIZE):
    #                  compressed_chunk = compressor.compress(chunk)
    #                  if compressed_chunk:
    #                      outfile.write(compressed_chunk)

    # Write any remaining data in the buffer

    #        return (ba_comp)

    def compress_torch_numpy_byte(self, data, lossy_compressed_type=None, lossy_compressed_factor=None):
        """
        Compresses torch.

        Parameters
        -------------------------------------
        data: torch.Tensor
                Torch data to compress.

        lossy_compressed_type: string
                ZipNN attribute lossy_compressed_type.
                Default is None.

        lossy_compressed_factor: int
                ZipNN attribute lossy_compressed_factor.
                Default is None.

        Returns
        -------------------------------------
        Byte array of compressed data.
        """
        is_print = 0
        is_review = 0
        bit_reorder = 0
        skip_split = 0
        # lossy_type = self.use_var(lossy_compressed_type, self.lossy_compressed_type)
        # lossy_type = EnumLossy.NONE if lossy_type is None else lossy_type
        # if lossy_type is not EnumLossy.NONE:
        #    lossy_factor = self.use_var(lossy_compressed_factor, self.lossy_compressed_factor)
        #    lossy_compress = self.lossy_compress(data, lossy_type, lossy_factor)

        if self.input_format == EnumFormat.BYTE.value:
            dtype_enum = ZipNNDtypeEnum.from_dtype(self.bytearray_dtype).code
            shape = None
        else:
            dtype_enum = ZipNNDtypeEnum.from_dtype(data.dtype).code
            shape = data.shape

        is_float = zipnn_is_floating_point(self.input_format, data, self.bytearray_dtype)
        if is_float:
            bit_reorder = 1
            if dtype_enum in (ZipNNDtypeEnum.FLOAT32.code, ZipNNDtypeEnum.FLOAT.code):
                byte_reorder = 220  # 8b1_10_11_100
                dtype_size = 32
            #            elif (dtype_enum == ZipNNDtypeEnum.BFLOAT16.code):
            elif dtype_enum == ZipNNDtypeEnum.BFLOAT16.code:
                byte_reorder = 10  # 8b01_010
                dtype_size = 16
                if self.input_format == EnumFormat.TORCH.value:
                    data = data.view(torch.uint16)
            elif dtype_enum in (ZipNNDtypeEnum.FLOAT16.code, ZipNNDtypeEnum.HALF.code):
                bit_reorder = 0
                byte_reorder = 10  # 8b01_010
                dtype_size = 16
            else:
                raise ValueError("Support only torch.dtype float32/bfloat16/float16")
        else:
            if dtype_enum == ZipNNDtypeEnum.UINT32.code and self.input_format == EnumFormat.NUMPY.value:
                max_val = np.max(data)
                dtype_size = 32
                if max_val < 256:  # truncate 3 bytes
                    byte_reorder = 1  # 8b0_00_00_001
                elif max_val < 65536:  # truncate 2 bytes
                    # It is faster to work with change the format to uint16 then to truncate out c implememtation
                    data = data.astype(np.uint16)
                    skip_split = 1  # use vanilla compression method
                    byte_reorder = 9  # 8b0_00_01_001
                elif max_val < 16777216:  # truncate 1 bytes
                    byte_reorder = 41  # 8b0_01_01_001
                else:  # not truncate anything use vanilla compression method
                    skip_split = 1  # use vanilla compression method
                    byte_reorder = 255  # all one
            else:
                raise ValueError("Support only uint32 with NumPy format")

        self._update_header_dtype(byte_reorder=byte_reorder, bit_reorder=bit_reorder, dtype_code=dtype_enum)

        is_review = 0

        start_time = time.time()

        if self.input_format == EnumFormat.TORCH.value:
            ba = data.numpy().tobytes()
        elif self.input_format == EnumFormat.NUMPY.value:
            ba = data.tobytes()
        elif self.input_format == EnumFormat.BYTE.value:
            ba = data
        else:
            raise ValueError("Unsupported input_format")

        if is_print:
            print("torch_func", time.time() - start_time)

        return self.compress_bin(
            ba=ba,
            byte_reorder=byte_reorder,
            bit_reorder=bit_reorder,
            is_review=is_review,
            is_float=is_float,
            dtype_size=dtype_size,
            shape=shape,
            skip_split=skip_split,
        )

    def lossy_compress(self, data, lossy_type, lossy_factor):
        """
        Handles lossy compression.

        Parameters
        -------------------------------------
        data:
                Data to compress.

        lossy_type: string
                ZipNN attribute lossy_compressed_type.

        lossy_factor: int
                ZipNN attribute lossy_compressed_factor.

        Returns
        -------------------------------------
        Data after lossy compression.
        """
        lossy_is_int = False
        if lossy_type == EnumLossy.INTEGER:
            bit_size, lossy_compressed_dtype = zipnn_get_dtype_bits(data.dtype)
            multiplier = 2**lossy_factor
            max_val = bit_size - 1 - lossy_factor
            data, lossy_is_int = zipnn_multiply_if_max_below(data, max_val, multiplier, lossy_compressed_dtype)
            self._update_header_lossy(lossy_type, lossy_factor, lossy_is_int)

        elif lossy_type == EnumLossy.UNSIGN:
            raise ValueError('lossy_compressed_type is "unsign" -> not implemented yet')
        else:
            raise ValueError(f"Unsupported lossy_compressed_type {lossy_type}")

        return data

    def compress_delta(self, delta_second_data, lossy_compressed_type, lossy_compressed_factor):
        """
        Handles delta compression.

        Parameters
        -------------------------------------
        delta_second_data: string
                Type of second data for the delta compression.

        lossy_type: string
                ZipNN attribute lossy_compressed_type.

        lossy_factor: int
                ZipNN attribute lossy_compressed_factor.

        Returns
        -------------------------------------
        Data after lossy compression.
        """
        raise ImportError("Not implemented Yet")

    #################
    # decompression #
    #################

    def decompress(self, data, decompress_cpu_gpu="cpu"):
        """
        Decompress is the ZipNN function used for decompression.

        Parameters
        -------------------------------------
        data: string
            The data to compress. It’s type can be one of the following options: ‘byte’, ‘torch’, ‘file’.
            If file, enter filename. Default is None.

        decompress_cpu_gpu: string
            Compression will be done by choice, in the CPU or GPU.
            Default is cpu.

        Returns
        -------------------------------------
        Returns the output of decompress_bin or decompress_read_file (depends on the type of the data compressed),
        which will be the compressed file, in the format chosen in the ZipNN class instance configuration.
        """
        mv_data = memoryview(data)
        comp_chunk_size = mv_data[13]  # 0 if no streaming > 127
        if self.input_format == EnumFormat.BYTE.value and comp_chunk_size > 127:
            decompressed_buffer = bytearray()
            offset = 0
            compressed_length = len(data)

            while offset < compressed_length:
                header = mv_data[offset : offset + 32]
                mid_chunk_len = int.from_bytes(header[24:32], byteorder="little") - 32
                chunk = mv_data[offset : offset + mid_chunk_len + 32]
                decompressed_chunk = self.decompress_bin(chunk)
                if decompressed_chunk:
                    decompressed_buffer.extend(decompressed_chunk)
                offset += mid_chunk_len + 32
            return decompressed_buffer
        return self.decompress_bin(data)

    def decompress_method(self, data):
        """
        Chooses decompression based on decompression method.

        Parameters
        -------------------------------------
        data: byte
                Data to decompress.

        Returns
        -------------------------------------
        Decompression of the data in the chosen method.
        """
        if self.method == EnumMethod.ZSTD.value or self.method == EnumMethod.AUTO.value:
            return self._zstd_decompress.decompress(data)
        if self.method == EnumMethod.LZ4.value:
            return lz4.frame.decompress(data)
        if self.method == EnumMethod.SNAPPY.value:
            return snappy.decompress(data)
        raise ValueError(f"Unsupported method {self.method}")

    def decompress_lossy(self, tensor, original_dtype):
        """
        Handles lossy decompression.

        Parameters
        -------------------------------------
        tensor: torch.Tensor
                The tensor data to decompress.

        original_dtype: string
                Original dtype value of the tensor.

        Returns
        -------------------------------------
        Tensor data after lossy decompression.
        """
        if self._lossy_is_int == 0:  # no need to transfer to integer from float
            tensor = tensor.view(original_dtype)
            return tensor
        # transfer from integer to float
        bit_size, int_dtype = zipnn_get_dtype_bits(original_dtype)
        tensor = tensor.view(int_dtype)
        lossy_factor = self.lossy_compressed_factor
        divisor = 2**lossy_factor
        decompress_tensor = zipnn_divide_int(tensor, divisor)
        return decompress_tensor

    def write_bin(self, ba_decom):
        """
        Writes decompressed data to file.

        Parameters
        -------------------------------------
        ba_decom: byte
                The data to write to the file.

        Returns
        -------------------------------------
        0 is succeed
        """
        with open(self.decompressed_file, "wb") as out_file_handler:
            out_file_handler.write(ba_decom)
        return 0

    def decompress_bin(self, ba_compress: bytes):
        """
        Decompresses byte data from either a byte array or a tensor.

        Parameters
        -------------------------------------
        ba_compress: byte
                Byte data to decompress.

        Returns
        -------------------------------------
        Returns a byte array of the decompressed data.
        """
        is_print = 0
        after_header = self._retrieve_header(ba_compress)

        dtype_size = 0  # Need to implement

        if (self.byte_reorder == 0b1_01_01_001 and dtype_size == 32) or (self.byte_reorder == 0b0_00_01_001 and dtype_size == 16):
            mv = memoryview(ba_compress[after_header:])
            ba_decom = self.decompress_method(mv[after_header:])
            if self.input_format == EnumFormat.BYTE.value:
                return ba_decom
            raise ValueError(f"Unsupported Torch with byte_reorder 0b1_01_01_001 or 0b0_00_01_001")
        else:
            float32 = 0
            bfloat16 = 0
            float16 = 0
            uint16 = 0
            uint32 = 0
            if self.dtype in (ZipNNDtypeEnum.FLOAT32.code, ZipNNDtypeEnum.FLOAT.code):
                groups = 4
                float32 = 1
            elif self.dtype == ZipNNDtypeEnum.BFLOAT16.code:
                groups = 2
                bfloat16 = 1
            elif self.dtype in (ZipNNDtypeEnum.FLOAT16.code, ZipNNDtypeEnum.HALF.code):
                groups = 2
                float16 = 1
            elif self.dtype == ZipNNDtypeEnum.UINT32.code:
                groups = 1
                uint32 = 1
            else:
                raise ValueError(f"Unsupported Dtype {self.dtype}")

            skip_combine = 0
            if self.input_format == EnumFormat.NUMPY.value and (self._byte_reorder in (9, 255)):
                skip_combine = 1

            ba_bg = []
            start_len = after_header + groups
            start_ba = [start_len + 8 * groups]
            end_ba = []
            if skip_combine == 0:
                if float32 or uint32:
                    for i in range(groups):
                        btime = time.time()
                        mv = memoryview(ba_compress)
                        is_comp = int.from_bytes(mv[after_header + i : after_header + i + 1], byteorder="little")
                        end_ba.append(int.from_bytes(mv[start_len + i * 8 : start_len + (i + 1) * 8 - 1], byteorder="little") + start_ba[i])
                        start_ba.append(end_ba[i])
                        if is_comp == 1:
                            ba_bg.append(self.decompress_method(ba_compress[start_ba[i] : end_ba[i]]))
                        else:
                            ba_bg.append(mv[start_ba[i] : end_ba[i]])
                        if is_print:
                            print(f"the time of this byte is: {time.time()-btime}")

                    if float32:
                        ba_decom = split_dtype.combine_dtype32(
                            ba_bg[0], ba_bg[1], ba_bg[2], ba_bg[3], self._bit_reorder, self._byte_reorder, self.threads
                        )
                    else:  # uint32_t
                        mv = memoryview(ba_compress)
                        ba_decom = split_dtype.combine_dtype32(
                            ba_bg[0], bytearray(0), bytearray(0), bytearray(0), self._bit_reorder, self._byte_reorder, self.threads
                        )
                elif bfloat16 or float16:
                    mv = memoryview(ba_compress)
                    ba_decom = split_dtype.combine_dtype16(
                        mv[after_header:], self._bit_reorder, self._byte_reorder, self.compression_chunk, self.original_len, self.threads
                    )
            else:
                ba_decom = ba_bg[0]

            if self.input_format == EnumFormat.BYTE.value:
                return ba_decom

            if self.input_format == EnumFormat.TORCH.value:
                if float32:
                    array = np.frombuffer(ba_decom, dtype=np.float32)
                    array = array.reshape(self.shape_bytes)
                    tensor = torch.from_numpy(array)
                elif bfloat16:
                    array = np.frombuffer(ba_decom, dtype=np.uint16)
                    array = array.reshape(self.shape_bytes)
                    tensor = torch.from_numpy(array)
                    tensor = tensor.view(torch.bfloat16)
                elif float16:
                    array = np.frombuffer(ba_decom, dtype=np.float16)
                    array = array.reshape(self.shape_bytes)
                    tensor = torch.tensor(array, dtype=torch.float16)
                return tensor

            if self.input_format == EnumFormat.NUMPY.value:
                if float32:
                    array = np.frombuffer(ba_decom, dtype=np.float32)
                elif float16:
                    array = np.frombuffer(ba_decom, dtype=np.float16)
                elif uint32:
                    if self._byte_reorder == 9:  # Truncate MSB, mid-high
                        array_uint16 = np.frombuffer(ba_decom, dtype=np.uint16)
                        array = array_uint16.astype(np.uint32)
                    else:
                        array = np.frombuffer(ba_decom, dtype=np.uint32)
                array = array.reshape(self.shape_bytes)
                return array

            raise ValueError(f"Unsupported input_format {self.input_format}")

    def decompress_read_file(self, data):
        """
        Decompresses data from file.

        Parameters
        -------------------------------------
        data: string
                The filename to decompress the data from.

        Returns
        -------------------------------------
        Byte array of the decompressed data.
        """
        filename = self.use_var(data, self.compressed_file)
        if not os.path.exists(filename):
            raise FileNotFoundError(f"The file at {filename} was not found.")
        with open(filename, "rb") as in_file_handler:
            ba = in_file_handler.read()
        return self.decompress_bin(ba)


#    def decompress_delta(self, base_path, delta_file):
#        return 0
