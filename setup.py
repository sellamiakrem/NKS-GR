# -*- coding: utf-8 -*-
"""
@author: Akrem Sellami
"""

from setuptools import setup, find_packages

setup(
    name="NKS-GR",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scikit-learn",
        "matplotlib",
    ],
    python_requires=">=3.8",
)
