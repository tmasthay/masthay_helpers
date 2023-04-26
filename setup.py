from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        "global_helpers_cython",
        sources=["global_helpers_cython.pyx"],
        extra_compile_args=["-O3"],
        extra_link_args=["-O3"],
    )
]

setup(
    name="Global Helpers Cython",
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
)
