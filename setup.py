from setuptools import setup, find_packages

setup(
    name="masthay_helpers",
    version="0.1",
    author="Tyler Masthay",
    description="Helper functions for repetitive and useful tasks",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=["numpy", "matplotlib"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

