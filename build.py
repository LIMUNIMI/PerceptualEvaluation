from distutils.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        "perceptual.alignment.cdist",
        ["perceptual/alignment/cdist.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'])
]


def build(setup_kwargs):
    setup_kwargs.update({
       'ext_modules': cythonize(
            extensions,
            compiler_directives={
                'language_level': "3",
                'embedsignature': True,
                'boundscheck': False,
                'wraparound': False
            })
       })
