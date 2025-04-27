import time
import os
import math
import multiprocessing
import numpy as np
from safetensors.torch import safe_open
import torch
import zipnn_core
from zipnn.util_header import EnumMethod, EnumFormat, EnumLossy
from zipnn.util_torch import (
    ZipNNDtypeEnum,
    zipnn_multiply_if_max_below,
    zipnn_get_dtype_bits,
    zipnn_divide_int,
    zipnn_pack_shape,
    zipnn_unpack_shape,
    zipnn_is_floating_point,
)
from zipnn.util_safetensors import (
    COMPRESSION_METHOD,
    COMPRESSED_DTYPE,
    get_compressed_tensors_metadata
)
from zipnn.util_patch import multi_process_patcher


class ZipNN:

    def __init__(
        self,
        method: str = "AUTO",
        input_format: str = "byte",
        bytearray_dtype: str = "bfloat16",
        is_monotonic: int = 0,
        threads: int = 0,
        compression_threshold=0.95,
        check_th_after_percent=10,
        byte_reorder: int = 0,
        reorder_signbit: int = 0,
        delta_compressed_type: str = 0,
        lossy_compressed_type: str = 0,
        lossy_compressed_factor=27,
        compression_chunk=256 * 1024,
        is_streaming: bool = False,
        streaming_chunk: int = 1024 * 1024,
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
                 The maximum threads for the compressio/decompression and for the byte/bit reorder.
                 If 0, the code decide according to the dataset len.
                 Default is the number of logical CPU threads

         compression_threshold: float
                 Save original buffer if not compress above the threshold (default value = 0.95).
                 Only relevant for a compression that uses byte grouping.
                 Default is 0.95.

         check_th_after_percent: int
                 Check the compression threshold after % from the number of chunk and stop compressing if not pass the compression_threshold.
                 Only relevant for a compression that uses byte grouping.
                 Default is 10[%]

         byte_reorder: int
                 Number of grouping.
                 4 Groups - Nu,ber for the group and zero for truncate.
                 [7] - Group 0/1 - 4'th Byte
                 [6-5] - Group 0/1/2 - 3'th Byte
                 [4-3] - Group 0/1/2/3 - 2'th Byte
                 [2-0] - Group 0/1/2/3/4 - 1'th Byte
                 for example:
                 bg16: two groups - 0_00_01_010 - decimal 10
                 fp32: four groups - 1_10_11_100 - decimal 220
                 int32: truncate two MSBs - 0_00_01_001 - decimal 9

         reorder_signbit: int
                 This reorder the bits of the float32 or bfloat16 to better compression.
                 If set to zero [default], auto decision according to the dtype
                 If set to 255 - no reorder_signbit.
                 If 16,32 - reorder_signbit for bfloat16 or float32 respectively
                 Default is 0 [Auto decision]

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
                 Cefault is 128KB

         is_streaming: bool
                 If true – signals compression is for a stream of data.
                 Default is False.

         streaming_chunk: int
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
                 Default is None.

         zstd_level: int
                 Compression level for ‘zstd’ compression.
                 Only relevant if method is ‘zstd’.
                 Default is 3.``

         lz4_compression_level: int
                 Compression level for ‘lz4’ compression.
                 Only relevant if method is ‘lz4’.
                 Default is 0.

         Returns
         -------------------------------------
         ZipNN class instance supporting a specific compression and decompression based on the input given.
        """

        self.method = EnumMethod(method).value
        self.input_format = EnumFormat(input_format).value
        self.bytearray_dtype = bytearray_dtype
        self.is_monotonic = is_monotonic
        # we've seen results deteriorate for threads > 16
        self.threads = threads or min(multiprocessing.cpu_count(), 16)
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

        if self.input_format != EnumFormat.BYTE.value and is_streaming:
            raise ValueError("Streaming is currently implemented only for bytes data type.")
        else:
            self.is_streaming = is_streaming

        if (streaming_chunk & (streaming_chunk - 1)) == 0:
            self.streaming_chunk = streaming_chunk
        else:
            raise ValueError("streaming_chunk must be a number that is a power of 2.")

        self.streaming_chunk = streaming_chunk

        self.input_file = input_file
        self.compressed_file = compressed_file
        self.decompressed_file = decompressed_file

        self.lz4_compression_level = lz4_compression_level

        self._version_major = 0
        self._version_minor = 5
        self._version_tiny = 3
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
        if self.method == EnumMethod.HUFFMAN.value or self.method == EnumMethod.AUTO.value:
            pass 
        elif self.method == EnumMethod.ZSTD.value:
            try: 
                import zstandard as zstd
            except ImportError as exc:
                raise ImportError("zstandard library is not installed. Please install it to use  zstandard compression: pip install zstandard ") from exc
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
    # [14] 1 Byte [is_streaming, streaming_chunk]
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
        Updates the shape of the data add to the and of the header
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
        self._header[9] = (
            0
            if self.delta_compressed_type is None
            else 1 if self.delta_compressed_type == "byte" else 2 if self.delta_compressed_type == "file" else 0
        )
        #        self._header[10] = self.lossy_compressed_type
        #        self._header[11] = self.lossy_compressed_factor
        #        self._header[12] = self._lossy_is_int
        if self.is_streaming:  # MSB is streaming, unsigned & is streaming
            self._header[13] = 128 + int(math.log(self.streaming_chunk, 2))
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
        header = mv[: self.header_length]
        if header[0:2].tobytes().decode("ascii") != "ZN":
            raise ValueError("Header should start with ZN")
        self.version_major = int(header[2])
        self.version_minor = int(header[3])
        self.version_tiny = int(header[4])
        self._byte_reorder = int(header[5])
        self._bit_reorder = int(header[6])
        self.method = int(header[7])
        self.input_format = int(header[8])
        self.delta_compressed_type = (
            0 if self._header[9] == 0 else "byte" if self._header[9] == 1 else "file" if self._header[9] == 2 else 0
        )
        self.lossy_compressed_type = int(header[10])
        self.lossy_compressed_factor = int(header[11])
        self._lossy_is_int = int(header[12])
        streaming_vals = int(header[13])
        if streaming_vals > 127:
            self.is_streaming = 1
            # self.streaming_chunk = 2 ** (128 - streaming_vals)
        else:
            self.is_streaming = 0
        self.compression_chunk = 2 ** header[14]
        self.dtype = int(header[15])
        self.original_len = int.from_bytes(header[16:24], byteorder="little")

        if self.input_format in (EnumFormat.TORCH.value, EnumFormat.NUMPY.value):
            self.shape_bytes, self._shape_size = zipnn_unpack_shape(mv[self.header_length :])
        return self.header_length + self._shape_size


    def __metadata__(self):
        """
        Retrieves metadata values of zipnn class instance in a dictionary and returns it.

        Parameters
        -------------------------------------

        Returns
        -------------------------------------
        A dictionary containing the header values.
        
        """
        header_dict = {
            "ZipNN version":str(self._version_major)+"."+str(self._version_minor)+"."+str(self._version_tiny),
            "Byte reorder": self.byte_reorder,
            "Bit reorder": self.reorder_signbit,
            "Method": self.method, 
            "Input format": self.input_format,
            "Data type": self.bytearray_dtype,
            "Is monotonic": self.is_monotonic,
            "Threads": self.threads,
            "Compression threshold": self.compression_threshold,
            "Check threshold after percent": self.check_th_after_percent,
            "Delta compressed type":self.delta_compressed_type,
            "Lossy compressed type":self.lossy_compressed_type,
            "Lossy compressed factor":self.lossy_compressed_factor,
            "Compression chunk":self.compression_chunk,
            "Is streaming":self.is_streaming,
            "Streaming chunk":self.streaming_chunk,
            "Input file path":self.input_file,
            "Compressedfile path":self.compressed_file,
            "Decompressed file path":self.decompressed_file
            #"Zstd level":self.zstd_level,
            #"Lz4 compression level":self.lz4_compression_level
        }
        

        print(header_dict)  # Print the dictionary for inspection
        return header_dict

    def __version__(self):
        """
        Retrieves Zipnn instance class version and prints it.

        Parameters
        -------------------------------------

        Returns
        -------------------------------------
        
        """

        print("ZipNN version: "+str(self._version_major)+"."+str(self._version_minor)+"."+str(self._version_tiny))  # Print the dictionary for inspection
        return


    def metadata(self,file,version=False):
        """
        Retrieves metadata values of compressed file and in a dictionary and returns it.

        Parameters
        -------------------------------------
        ba_compress: byte
            Header data compressed to byte array.

        Returns
        -------------------------------------
        A dictionary containing the header values.
        """
        if type(file)==str:
            with open(file, "rb") as f:
                header_data = f.read(self.header_length)
                mv = memoryview(header_data)
                header = mv[: self.header_length]
        else:
            mv = memoryview(file)
            header = mv[: self.header_length]

        if header[0:2].tobytes().decode("ascii") != "ZN":
            raise ValueError("Header should start with ZN")

        if version:
            print("ZipNN version: "+str(header[2])+"."+str(header[3])+"."+str(header[4]))
            return
        
        header_dict = {
            "zipnn version": str(header[2])+"."+str(header[3])+"."+str(header[4]),
            "byte_reorder": int(header[5]),
            "bit_reorder": int(header[6]),
            "method": EnumMethod(int(header[7])).name if int(header[7]) in EnumMethod._value2member_map_ else "UNKNOWN", 
            "input_format": EnumFormat(int(header[8])).name if int(header[8]) in EnumMethod._value2member_map_ else "UNKNOWN",
            "delta_compressed_type": (
                0 if header[9] == 0 else "byte" if header[9] == 1 else "file" if header[9] == 2 else 0
            ),
            "lossy_compressed_type": EnumLossy(int(header[10])).name if int(header[10]) in EnumMethod._value2member_map_ else "NONE",
            "lossy_compressed_factor": int(header[11]),
            "lossy_is_int": int(header[12]), #
            "is_streaming": True if int(header[13]) > 127 else False,
            "compression_chunk": f"{2 ** header[14]} Bytes",
            "dtype" :ZipNNDtypeEnum.from_code(int(header[15])),
            "original_len": f"{int.from_bytes(header[16:24], byteorder='little')} Bytes"

        }
        if int(header[8]) in (EnumFormat.TORCH.value, EnumFormat.NUMPY.value):
            shape_bytes, shape_size = zipnn_unpack_shape(mv[self.header_length:])
            header_dict["shape_bytes"] = shape_bytes
            header_dict["shape_size"] = shape_size
            return_size = self.header_length + shape_size
        else:
            return_size = self.header_length

        print(header_dict)  # Print the dictionary for inspection
        return header_dict


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
        if self.delta_compressed_type == "byte":
            if len(data) != len(delta_second_data):
                raise ValueError("Length of delta file has to match the length of the original file.")
        elif self.delta_compressed_type == "file":
            try:
                with open(delta_second_data, "rb") as file:
                    file_data = file.read()
                delta_second_data = file_data
            except Exception:
                raise FileNotFoundError("Encountered an error when reading the delta file")
            if len(data) != len(file_data):
                raise ValueError("Length of delta file has to match the length of the original file.")
            delta_second_data = file_data
        else:  # self.delta_compressed_type="0"
            if delta_second_data != None:
                raise ValueError("ZipNN isn't set for delta compression, but delta_second_data is not null.")

        if self.is_streaming and self.input_format == EnumFormat.BYTE.value:
            mv_data = memoryview(data)
            if delta_second_data:
                mv_delta = memoryview(delta_second_data)
            CHUNK_SIZE = self.streaming_chunk
            # Compression into bytearray
            compressed_buffer = bytearray()
            remaining_bytes = len(data)
            offset = 0

            while remaining_bytes > 0:
                chunk_size = min(CHUNK_SIZE, remaining_bytes)
                chunk = mv_data[offset : offset + chunk_size]
                if delta_second_data:
                    chunk_delta = mv_delta[offset : offset + chunk_size]
                    array1 = np.frombuffer(chunk, dtype=np.uint8)
                    array2 = np.frombuffer(chunk_delta, dtype=np.uint8)
                    chunk = np.bitwise_xor(array1, array2).tobytes()
                compressed_chunk = self.compress_torch_numpy_byte(chunk, lossy_compressed_type, lossy_compressed_factor)
                if compressed_chunk:
                    compressed_buffer.extend(compressed_chunk)
                offset += chunk_size
                remaining_bytes -= chunk_size
            return compressed_buffer
        else:
            if delta_second_data:
                array1 = np.frombuffer(data, dtype=np.uint8)
                array2 = np.frombuffer(delta_second_data, dtype=np.uint8)
                data = np.bitwise_xor(array1, array2).tobytes()
            #        if self.delta_compressed_type is not None:
            #            return self.compress_delta(data, delta_second_data, lossy_compressed_type, lossy_compressed_factor)
            return self.compress_torch_numpy_byte(data, lossy_compressed_type, lossy_compressed_factor)

    def compress_method(self, data: memoryview):
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
        if self.method == EnumMethod.HUFFMAN.value or self.method == EnumMethod.AUTO.value:
            pass 
        if self.method in (EnumMethod.ZSTD.value, EnumMethod.AUTO.value):
            return self._zstd_compress.compress(data)

        if self.method == EnumMethod.LZ4.value:
            return lz4.frame.compress(data)

        if self.method == EnumMethod.SNAPPY.value:
            return snappy.compress(data)
        raise ValueError(f"Unsupported method {self.method}")

    def compress_bin(
        self,
        ba: memoryview,
        bit_reorder: int,
        byte_reorder: int,
        is_review: int,
        is_float: int,
        dtype_size: int,
        num_buf: int,
        shape,
        skip_split: bool,
    ):
        """
        Compresses byte data.

        Parameters
        -------------------------------------
        ba: memoryview
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

            if is_print:
                start_time = time.time()
            self._update_header_original_len(len(ba))
            if self.input_format in (EnumFormat.TORCH.value, EnumFormat.NUMPY.value):
                self._update_data_shape(shape)
            python_header = self._header + self._ext_header
            ba_saved = bytearray(ba)
            ba_comp = zipnn_core.zipnn_core(
                python_header,
                ba,
                num_buf,
                bit_reorder,
                byte_reorder,
                is_review,
                self.compression_chunk if num_buf!=1 else min(128*1024,self.compression_chunk), # Huffman compression is limited to a 128K buffer; therefore, we restrict it to 128K in the case of FP8.
                self.compression_threshold,
                self.check_th_after_percent,
                self.threads,
            )
            #
            #ba_decom = zipnn_core.combine_dtype(
            #    ba_comp[len(python_header):],
            #    num_buf,
            #    bit_reorder,
            #    byte_reorder,
            #    self.compression_chunk,
            #    len(ba),
            #    self.threads,
            #    )
            #print("Are the original and decompressed byte strings the same [BYTE]? ", ba_decom[:32] == ba_saved[:32])
            #print(f"ratio: {len(ba_comp)/len(ba_saved)}")
            #
            if is_print:
                print("aggregate output bin ", time.time() - start_time)
        if is_print:
            print("total compression ", time.time() - stime)
            print(f"len ba-comp {len(ba_comp)}")
            print(f"len ba {len(ba)}")
            print("compress_bin_time ", time.time() - compress_bin_time)
        return ba_comp

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
            if dtype_enum in (ZipNNDtypeEnum.FLOAT8_E4M3FN.code, ZipNNDtypeEnum.FLOAT8_E5M2.code):
                # FP8 handeling
                num_buf=1
                dtype_size=8
                byte_reorder = 10
                if not (self.input_format == EnumFormat.BYTE.value):
                    data = data.view(torch.uint8)
                # print(data[:16])
            elif dtype_enum in (ZipNNDtypeEnum.FLOAT32.code, ZipNNDtypeEnum.FLOAT.code):
                byte_reorder = 220  # 8b1_10_11_100
                dtype_size = 32
                num_buf = 4
            #            elif (dtype_enum == ZipNNDtypeEnum.BFLOAT16.code):
            elif dtype_enum == ZipNNDtypeEnum.BFLOAT16.code:
                byte_reorder = 10  # 8b01_010
                dtype_size = 16
                num_buf = 2
                if self.input_format == EnumFormat.TORCH.value:
                    data = data.view(torch.uint16)
            elif dtype_enum in (ZipNNDtypeEnum.FLOAT16.code, ZipNNDtypeEnum.HALF.code, ZipNNDtypeEnum.FLOAT8_E4M3FN.code, ZipNNDtypeEnum.FLOAT8_E5M2):
                bit_reorder = 0
                byte_reorder = 10  # 8b01_010
                dtype_size = 16
                num_buf = 2
            else:
                raise ValueError("Support only torch.dtype float32/bfloat16/float16")
        else:
            if dtype_enum == ZipNNDtypeEnum.UINT32.code and self.input_format == EnumFormat.NUMPY.value:
                raise ValueError("Not support uint32 with NumPy format")
                max_val = np.max(data)
                dtype_size = 32
                num_buf = 4
                if max_val < 256:  # truncate 3 bytes
                    byte_reorder = 1  # 8b0_00_00_001
                elif max_val < 65536:  # truncate 2 bytes
                    # It is faster to work with change the format to uint16 then to truncate out c implementation
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
            if dtype_enum in (ZipNNDtypeEnum.FLOAT8_E4M3FN.code, ZipNNDtypeEnum.FLOAT8_E5M2):
                data = data.view(torch.uint8)
            ba = memoryview(data.contiguous().view(-1).numpy()).cast("B")
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
            num_buf=num_buf,
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

    def decompress(self, data, decompress_cpu_gpu="cpu", delta_second_data=None):
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
        if self.delta_compressed_type == "byte":
            if delta_second_data is None:
                raise ValueError("delta_second_data is None or not set for delta copression")
        elif self.delta_compressed_type == "file":
            try:
                with open(delta_second_data, "rb") as file:
                    file_data = file.read()
                delta_second_data = file_data
            except Exception:
                raise FileNotFoundError("Encountered an error when reading the delta file")
        else:  # self.delta_compressed_type==0
            if delta_second_data != None:
                raise ValueError("ZipNN isn't set for delta compression, but delta_second_data is not null.")

        mv_data = memoryview(data)

        was_data_delta_compressed = mv_data[9]
        if was_data_delta_compressed == 0 and self.delta_compressed_type != 0:
            raise ValueError("The data wasn't compressed using delta compression and you're trying to delta-decompress it.")
        if was_data_delta_compressed != 0 and self.delta_compressed_type == 0:
            raise ValueError("The data was compressed using delta compression and you're trying to decompress it normally.")
        if delta_second_data:
            mv_delta = memoryview(delta_second_data)

        comp_chunk_size = mv_data[13]  # 0 if no streaming > 127
        if self.input_format == EnumFormat.BYTE.value and comp_chunk_size > 127:  # xor inside streaming
            decompressed_buffer = bytearray()
            offset = 0
            compressed_length = len(data)
            offset_delta = 0
            while offset < compressed_length:
                header = mv_data[offset : offset + 32]
                mid_chunk_len = int.from_bytes(header[24:32], byteorder="little") - 32
                chunk = mv_data[offset : offset + mid_chunk_len + 32]
                decompressed_chunk = self.decompress_bin(chunk)
                if decompressed_chunk:
                    if delta_second_data:
                        if offset_delta + len(decompressed_chunk) > len(mv_delta):
                            raise ValueError("Length of delta file has to match the length of the decompressed file.")
                        chunk_delta = mv_delta[offset_delta : offset_delta + len(decompressed_chunk)]
                        array1 = np.frombuffer(decompressed_chunk, dtype=np.uint8)
                        array2 = np.frombuffer(chunk_delta, dtype=np.uint8)
                        decompressed_chunk = np.bitwise_xor(array1, array2).tobytes()
                        offset_delta += len(decompressed_chunk)
                    decompressed_buffer.extend(decompressed_chunk)
                offset += mid_chunk_len + 32
            if delta_second_data and offset_delta != len(mv_delta):
                raise ValueError("Length of delta file has to match the length of the decompressed file.")
            return decompressed_buffer

        if delta_second_data:
            decompressed_buffer = self.decompress_bin(data)
            if len(decompressed_buffer) != len(delta_second_data):
                raise ValueError("Length of delta file has to match the length of the decompressed file.")
            array1 = np.frombuffer(decompressed_buffer, dtype=np.uint8)
            array2 = np.frombuffer(delta_second_data, dtype=np.uint8)
            final_data = np.bitwise_xor(array1, array2).tobytes()
            return final_data
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
            float8 = 0
            if self.dtype in (ZipNNDtypeEnum.FLOAT8_E4M3FN.code, ZipNNDtypeEnum.FLOAT8_E5M2.code):
                # FP8 handeling
                groups = 1
                float8 = 1
            elif self.dtype in (ZipNNDtypeEnum.FLOAT32.code, ZipNNDtypeEnum.FLOAT.code):
                groups = 4
                float32 = 1
            elif self.dtype == ZipNNDtypeEnum.BFLOAT16.code:
                groups = 2
                bfloat16 = 1
            elif self.dtype in (ZipNNDtypeEnum.FLOAT16.code, ZipNNDtypeEnum.HALF.code):
                groups = 2
                float16 = 1
            elif self.dtype in (ZipNNDtypeEnum.FLOAT8_E4M3FN.code, ZipNNDtypeEnum.FLOAT8_E5M2.code):
                groups = 2
                float8 = 1
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
                num_buf = 4
                if uint32:
                    raise ValueError("Unsupported uinit32 in this version yet! please try version 0.1.1")
                elif bfloat16 or float16:
                    num_buf = 2
                elif float8:
                    # FP8 handeling
                    num_buf = 1
                mv = memoryview(ba_compress)
                ba_decom = zipnn_core.combine_dtype(
                    mv[after_header:],
                    num_buf,
                    self._bit_reorder,
                    self._byte_reorder,
                    self.compression_chunk if num_buf!=1 else min(128*1024,self.compression_chunk),
                    self.original_len,
                    self.threads,
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
                    tensor = torch.from_numpy(array)
                elif float8:
                    # FP8 handeling
                    array = np.frombuffer(ba_decom, dtype=np.uint8) 
                    new_shape = tuple(dim for dim in self.shape_bytes)
                    array = array.reshape(new_shape)
                    tensor = torch.from_numpy(array)
                    if self.dtype==ZipNNDtypeEnum.FLOAT8_E5M2.code:
                        tensor = tensor.view(torch.float8_e5m2)
                    else:
                        tensor = tensor.view(torch.float8_e4m3fn)
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


def zipnn_hf(replace_local_file: bool = False):
    """
    Plugin for the Hugging Face Transformers library to use ZipNN compression.

    Parameters
    -------------------------------------
    replace_local_file: bool
        If True, replace the local file with the decompressed file and deletes the decompressed file.

    Returns
    -------------------------------------
    None.
    """
    try:
        from transformers import modeling_utils
        from typing import Union, Optional, Dict
        from transformers.configuration_utils import PretrainedConfig
        from transformers.utils import (
            FLAX_WEIGHTS_NAME,
            SAFE_WEIGHTS_INDEX_NAME,
            SAFE_WEIGHTS_NAME,
            TF2_WEIGHTS_NAME,
            TF_WEIGHTS_NAME,
            WEIGHTS_INDEX_NAME,
            WEIGHTS_NAME,
            cached_file,
        )
        import transformers.modeling_utils
        from transformers.modeling_utils import _add_variant, PreTrainedModel, is_deepspeed_zero3_enabled, is_fsdp_enabled, is_torch_greater_or_equal, is_zipfile, is_local_dist_rank_0
        from safetensors.torch import load
        import json
        from struct import unpack
        from packaging import version
        from io import BytesIO

    except ImportError as exc:
        raise ImportError("Hugging Face Transformers library is not installed. Please install it to use ZipNN compression.") from exc

    from typing import Union
    import transformers

    # Save the original load_state_dict method
    original_load_state_dict = modeling_utils.load_state_dict

    # Check the version of transformers
    transformers_version = transformers.__version__

    def decompress_znn(checkpoint_file: Union[str, os.PathLike], replace_local_file: bool = False, is_quantized: bool = False, map_location: Optional[Union[str, torch.device]] = None, weights_only: bool = True):
        if checkpoint_file.endswith(".znn"):
            print(f"Decompressing {checkpoint_file.split('/')[-1]}")

            ### Loading with buffer only supports safetensors for now
            # if not replace_local_file and not checkpoint_file.endswith(".safetensors.znn"):
            #     print("\033[91mZipNN only supports .safetensors.znn for now. Saving decompressed file locally.\033[0m")
            #     replace_local_file = True

            output_file = checkpoint_file.replace(".znn", "")
            snapshot_path = os.path.dirname(checkpoint_file)
            d_data = b""
            if not os.path.exists(output_file):
                znn = ZipNN(is_streaming=True)
                with open(checkpoint_file, "rb") as infile:
                    chunk = infile.read()
                    d_data += znn.decompress(chunk)

                    ### Save the decompressed file
                    if replace_local_file:
                        with open(output_file, "wb") as outfile:
                            outfile.write(d_data)
                            
                ### Replace the local file with the decompressed file
                if replace_local_file:
                    blob_name = os.path.join(snapshot_path, os.readlink(checkpoint_file))
                    os.rename(output_file, blob_name)
                    os.symlink(blob_name, output_file)
            else:
                print(f"Decompressed file already exists at {output_file}")
                with open(output_file, "rb") as infile:
                    d_data = infile.read()

            ### Remove the compressed file and change the index name
            if replace_local_file:
                os.remove(checkpoint_file)
                checkpoint_file = output_file

                # Change index name to the decompressed file
                if os.path.exists(os.path.join(snapshot_path, SAFE_WEIGHTS_INDEX_NAME)):
                    file_name = os.path.basename(output_file)
                    blob_name = os.path.join(snapshot_path, os.readlink(os.path.join(snapshot_path, SAFE_WEIGHTS_INDEX_NAME)))
                    replace_in_file(file_path=blob_name, old=f"{file_name}.znn", new=f"{file_name}")

                elif os.path.exists(os.path.join(snapshot_path, WEIGHTS_INDEX_NAME)):
                    file_name = os.path.basename(output_file)
                    blob_name = os.path.join(snapshot_path, os.readlink(os.path.join(snapshot_path, WEIGHTS_INDEX_NAME)))
                    replace_in_file(file_path=blob_name, old=f"{file_name}.znn", new=f"{file_name}")
            elif d_data:
                if checkpoint_file.endswith(".safetensors.znn"):
                    length_of_header = unpack('<Q', d_data[:8])[0]
                    header_data = d_data[8:8 + length_of_header]
                    header = json.loads(header_data)

                    # Check safetensors metadata
                    metadata = header.get("__metadata__", {})
                    if metadata.get("format") not in ["pt", "tf", "flax", "mlx"]:
                        raise OSError(
                            f"The safetensors archive passed at {checkpoint_file} does not contain the valid metadata. Make sure "
                            "you save your model with the `save_pretrained` method."
                        )
                    return load(d_data)
                try:
                    if map_location is None:
                        if (
                            (
                                is_deepspeed_zero3_enabled()
                                and torch.distributed.is_initialized()
                                and torch.distributed.get_rank() > 0
                            )
                            or (is_fsdp_enabled() and not is_local_dist_rank_0())
                        ) and not is_quantized:
                            map_location = "meta"
                        else:
                            map_location = "cpu"
                    extra_args = {}
                    # mmap can only be used with files serialized with zipfile-based format.
                    if (
                        isinstance(checkpoint_file, str)
                        and map_location != "meta"
                        and version.parse(torch.__version__) >= version.parse("2.1.0")
                        and is_zipfile(checkpoint_file)
                    ):
                        extra_args = {"mmap": True}
                    weights_only_kwarg = {"weights_only": weights_only} if is_torch_greater_or_equal("1.13") else {}
                    return torch.load(
                        BytesIO(d_data),
                        map_location=map_location,
                        **weights_only_kwarg,
                        **extra_args,
                    )
                except Exception as e:
                    try:
                        with open(checkpoint_file) as f:
                            if f.read(7) == "version":
                                raise OSError(
                                    "You seem to have cloned a repository without having git-lfs installed. Please install "
                                    "git-lfs and run `git lfs install` followed by `git lfs pull` in the folder "
                                    "you cloned."
                                )
                            else:
                                raise ValueError(
                                    f"Unable to locate the file {checkpoint_file} which is necessary to load this pretrained "
                                    "model. Make sure you have saved the model properly."
                                ) from e
                    except (UnicodeDecodeError, ValueError):
                        raise OSError(
                            f"Unable to load weights from pytorch checkpoint file for '{checkpoint_file}' "
                            f"at '{checkpoint_file}'. "
                            "If you tried to load a PyTorch model from a TF 2.0 checkpoint, please set from_tf=True."
                        )
                
    if transformers_version > "4.45.2":
        # Define a monkey-patched version of load_state_dict
        def custom_load_state_dict(checkpoint_file: Union[str, os.PathLike], is_quantized: bool = False, map_location: Optional[Union[str, torch.device]] = None, weights_only: bool = True):
            # Decompress the checkpoint file
            result = decompress_znn(checkpoint_file, replace_local_file, is_quantized=is_quantized, map_location=map_location, weights_only=weights_only)
            if result:
                return result

            if not os.path.exists(checkpoint_file) and os.path.exists(checkpoint_file.replace(".znn", "")):
                checkpoint_file = checkpoint_file.replace(".znn", "")

            # Call the original load_state_dict method
            return original_load_state_dict(checkpoint_file, is_quantized, map_location, weights_only)
    else:
        # Define a monkey-patched version of load_state_dict
        def custom_load_state_dict(checkpoint_file: Union[str, os.PathLike], is_quantized: bool = False):
            # Decompress the checkpoint file
            result = decompress_znn(checkpoint_file, replace_local_file, is_quantized=is_quantized)
            if result:
                return result
            
            if not os.path.exists(checkpoint_file) and os.path.exists(checkpoint_file.replace(".znn", "")):
                checkpoint_file = checkpoint_file.replace(".znn", "")

            # Call the original load_state_dict method
            return original_load_state_dict(checkpoint_file, is_quantized)
    
    # Monkey patch the load_state_dict method in the transformers library
    modeling_utils.load_state_dict = custom_load_state_dict
    

    # save original from_pretrained
    original_from_pretrained = PreTrainedModel.from_pretrained

    # Found paths to check
    found_paths = []

    # class CustomPreTrainedModel(PreTrainedModel):
    def custom_from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: bool = None,
        **kwargs,
    ):

        subfolder = kwargs.get("subfolder", "")
        variant = kwargs.get("variant", None)
        proxies = kwargs.get("proxies", None)
        resume_download = kwargs.get("resume_download", None)
        commit_hash = kwargs.get("_commit_hash", None)
        from_pipeline = kwargs.get("_from_pipeline", None)
        from_auto_class = kwargs.get("_from_auto", False)

        # Skipping user_agent["quant"]! Could be a problem
        user_agent = {"file_type": "model", "framework": "pytorch", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        test_paths_org = [
            TF_WEIGHTS_NAME + ".index",
            TF2_WEIGHTS_NAME,
            FLAX_WEIGHTS_NAME,
            _add_variant(SAFE_WEIGHTS_NAME, variant),
            _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant),
            _add_variant(WEIGHTS_NAME, variant),
            _add_variant(WEIGHTS_INDEX_NAME, variant),
            FLAX_WEIGHTS_NAME,
            pretrained_model_name_or_path,
            pretrained_model_name_or_path + ".index",
        ]

        test_paths = [path + ".znn" for path in test_paths_org]

        cached_file_kwargs = {
            "cache_dir": cache_dir,
            "force_download": force_download,
            "proxies": proxies,
            "resume_download": resume_download,
            "local_files_only": local_files_only,
            "token": token,
            "user_agent": user_agent,
            "revision": revision,
            "subfolder": subfolder,
            "_raise_exceptions_for_gated_repo": False,
            "_raise_exceptions_for_missing_entries": False,
            "_commit_hash": commit_hash,
        }

        for i, filename in enumerate(test_paths):
            resolved_archive_file = cached_file(pretrained_model_name_or_path, filename, **cached_file_kwargs)
            if resolved_archive_file is not None:
                if not replace_local_file:
                    found_paths.append(test_paths_org[i])
                else:
                    print(f"Decompressing {resolved_archive_file.split('/')[-1]}")
                    output_file = resolved_archive_file.replace(".znn", "")
                    if not os.path.exists(output_file):
                        znn = ZipNN(is_streaming=True)
                        with open(resolved_archive_file, "rb") as infile, open(output_file, "wb") as outfile:
                            d_data = b""
                            chunk = infile.read()
                            d_data += znn.decompress(chunk)
                            outfile.write(d_data)
                        snapshot_path = os.path.dirname(resolved_archive_file)
                        blob_name = os.path.join(snapshot_path, os.readlink(resolved_archive_file))
                        os.rename(output_file, blob_name)
                        os.symlink(blob_name, output_file)
                    os.remove(resolved_archive_file)
        # pack config, cache_dir, etc. into kwargs
        kwargs.update(
            {
                "config": config,
                "cache_dir": cache_dir,
                "ignore_mismatched_sizes": ignore_mismatched_sizes,
                "force_download": force_download,
                "local_files_only": local_files_only,
                "token": token,
                "revision": revision,
                "use_safetensors": use_safetensors,
            }
        )

        # Call the original from_pretrained method with the updated kwargs
        return original_from_pretrained.__func__(
            cls,
            pretrained_model_name_or_path,
            *model_args,
            **kwargs,
        )

    # Monkey patch the from_pretrained method in the transformers library
    PreTrainedModel.from_pretrained = classmethod(custom_from_pretrained)

    # Monkey patch chached_file to add .znn extension if filename inputed
    original_cached_file = modeling_utils.cached_file

    def custom_cached_file(
        path_or_repo_id: Union[str, os.PathLike],
        filename: str,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        force_download: bool = False,
        resume_download: Optional[bool] = None,
        proxies: Optional[Dict[str, str]] = None,
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        local_files_only: bool = False,
        subfolder: str = "",
        repo_type: Optional[str] = None,
        user_agent: Optional[Union[str, Dict[str, str]]] = None,
        _raise_exceptions_for_gated_repo: bool = True,
        _raise_exceptions_for_missing_entries: bool = True,
        _raise_exceptions_for_connection_errors: bool = True,
        _commit_hash: Optional[str] = None,
        **deprecated_kwargs,
    ):
        if filename in found_paths:
            filename = filename + ".znn"
        return original_cached_file(
            path_or_repo_id,
            filename,
            cache_dir,
            force_download,
            resume_download,
            proxies,
            token,
            revision,
            local_files_only,
            subfolder,
            repo_type,
            user_agent,
            _raise_exceptions_for_gated_repo,
            _raise_exceptions_for_missing_entries,
            _raise_exceptions_for_connection_errors,
            _commit_hash,
            **deprecated_kwargs,
        )
    
    modeling_utils.cached_file = custom_cached_file


def replace_in_file(file_path, old: str, new: str) -> None:
    """Given a file_path, replace all occurrences of `old` with `new` inplace."""

    with open(file_path, "r") as file:
        file_data = file.read()

    file_data = file_data.replace(old, new)

    with open(file_path, "w") as file:
        file.write(file_data)


#    def decompress_delta(self, base_path, delta_file):
#        return 0


def decompress_safetensors_tensor(tensor: torch.tensor) -> torch.tensor:
    """
    decompress a tensor from a compressed safetensors file.
    """
    znn = ZipNN(input_format="torch", bytearray_dtype=COMPRESSED_DTYPE, method=COMPRESSION_METHOD)
    return znn.decompress(tensor.contiguous().numpy())


class SafeOpen:
    """
    safetensors safe_open wrapper class for injecting tensor decompression support.
    """

    def __init__(self, filename, framework, device="cpu"):
        self._f = safe_open(filename, framework, device)
        self.compressed_tensors_metadata = get_compressed_tensors_metadata(self._f.metadata())

    def get_tensor(self, name):
        """
        gets a (possibly compressed) tensor from the safetensors file.
        """
        if name not in self.compressed_tensors_metadata:
            return self._f.get_tensor(name)
        return decompress_safetensors_tensor(self._f.get_tensor(name))

    def get_slice(self, name):
        """
        gets a alice of a (possibly compressed) tensor from the safetensors file.

        Compressed tensors are currently unsupported by this function.
        """
        if name not in self.compressed_tensors_metadata:
            return self._f.get_slice(name)
        return NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self._f.__exit__(exc_type, exc_value, traceback)

    def __getattr__(self, name):
        return getattr(self._f, name)


def _zipnn_safetensors():
    """
    single process patching of safetensors library to use ZipNN compression.
    """
    import safetensors.torch

    safetensors.torch.safe_open = SafeOpen


def zipnn_safetensors():
    """
    Plugin for the safetensors library to use ZipNN compression.
    """

    multi_process_patcher(_zipnn_safetensors)
