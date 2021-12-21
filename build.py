from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("asmd.asmd.audioscoredataset",
              ["asmd/asmd/audioscoredataset.pyx"],
              include_dirs=[numpy.get_include()]),
    Extension("asmd.asmd.convert_from_file",
              ["asmd/asmd/convert_from_file.pyx"]),
    Extension("asmd.asmd.conversion_tool", ["asmd/asmd/conversion_tool.pyx"]),
    Extension("asmd.asmd.utils", ["asmd/asmd/utils.pyx"]),
    Extension("perceptual.alignment.cdist", ["perceptual/alignment/cdist.pyx"],
              extra_compile_args=['-fopenmp'],
              extra_link_args=['-fopenmp'])
]


def build(setup_kwargs):
    setup_kwargs.update({
        'ext_modules':
        cythonize(extensions,
                  compiler_directives={
                      'language_level': "3",
                      'embedsignature': True,
                  })
    })
