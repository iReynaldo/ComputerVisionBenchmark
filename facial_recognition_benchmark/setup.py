"""
Setup script for the facial recognition benchmark module.
"""

from setuptools import setup, find_packages

setup(
    name="facial_recognition_benchmark",
    version="1.0.0",
    description="A comprehensive benchmark suite for facial recognition applications",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="CogWorks",
    author_email="cogworks@example.com",
    url="https://github.com/CogWorksBWSI/facial_recognition_benchmark",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
        ],
        "image": [
            "scikit-image>=0.18.0",
        ],
        "full": [
            "facenet_models",
            "scikit-image>=0.18.0",
            "networkx>=2.0",
            "matplotlib>=3.3.0",
        ],
    },
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
)
