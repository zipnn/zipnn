from setuptools import setup, Extension

module = Extension('split_dtype',
                    sources=['split_dtype_module.c','split_dtype32.c', 'split_dtype16.c'],
                    extra_compile_args=['-O3'],
                    extra_link_args=['-O3']
                   )

setup(name='split_dtype',
      version='1.0',
      description='Split a bytearray into four buffers',
      ext_modules=[module])
