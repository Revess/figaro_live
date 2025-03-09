from setuptools import setup
import os

setup(
    name="Figaro_Live",
    version="1.0.0",
    author="Bas Maat",
    author_email="bjmaat@gmail.com",
    description="",
    packages=['figaro'],
    install_requires=[
        "numpy~=1.26.3",
        "pandas~=2.2.0",
        "pretty-midi==0.2.10",
        "pytorch-lightning~=2.1.3",
        "scikit-learn~=1.4.0",
        "scipy~=1.12.0",
        "torch~=2.1.2",
        "torchtext~=0.16.2",
        "transformers~=4.37.0",
        "ipykernel",
        "mido",
        "python-rtmidi",
        "py_midicsv"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Replace with your license
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.11',  # Minimum Python version
    include_package_data=True,  # Include additional files specified in MANIFEST.in
)