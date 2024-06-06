from setuptools import find_packages, setup

# https://pypi.org/project/masthay-helpers/

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="masthay_helpers",
    version="0.2.100",
    author="Tyler Masthay",
    description="Helper functions for repetitive and useful tasks",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
