from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        "helpers.global_helpers_cython",
        sources=["helpers/global_helpers_cython.pyx"],
        extra_compile_args=["-O3"],
        extra_link_args=["-O3"],
    )
]

setup(
    name="helpers",
    version="0.1",
    author="Tyler Masthay",
    description="Helper functions for repetitive and useful tasks",
    packages=find_packages(),
    install_requires=["numpy", "matplotlib"],
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

