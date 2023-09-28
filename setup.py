from setuptools import setup, find_packages

# https://pypi.org/project/masthay-helpers/

setup(
    name="masthay_helpers",
    version="0.2.16",
    author="Tyler Masthay",
    description="Helper functions for repetitive and useful tasks",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=["numpy", "matplotlib"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
