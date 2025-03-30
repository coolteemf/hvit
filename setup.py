#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import setup, find_packages

# The text of the README file
with open("README.md", encoding="utf-8") as f:
    README = f.read()

# This setup.py is compatible with the pyproject.toml configuration
# and serves as a fallback for older pip versions

setup(
    name="hvit",
    version="0.1.0",
    description="A Hierarchical Vision Transformer for Deformable Image Registration",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Morteza Ghahremani, Mohammad Khateri, Bailiang Jian, Benedikt Wiestler, Ehsan Adeli, Christian Wachinger",
    python_requires=">=3.10",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    keywords=["deep-learning", "vision-transformer", "image-registration", "medical-imaging"],
    package_dir={"": "src"},
    packages=find_packages(where="src"),  # This will find the 'hvit' package inside src/hvit
    install_requires=[
        "einops",
        "timm",
        "lightning",
        "wandb",
        "monai",
        "gitpython",
    ],
    extras_require={
        "dev": [
            "black",
            "flake8",
            "pytest",
        ],
        "torch": [
            "torch>=1.12.0",
            "torchvision",
            "torchaudio",
        ],
    },
    url="https://github.com/mogvision/hvit",
    project_urls={
        "Paper": "https://openaccess.thecvf.com/content/CVPR2024/html/Ghahremani_H-ViT_A_Hierarchical_Vision_Transformer_for_Deformable_Image_Registration_CVPR_2024_paper.html",
    },
)

