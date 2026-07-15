"""
Setup script for the face recognition application.
"""

from setuptools import setup, find_packages

setup(
    name="face_recognition",
    version="1.0.0",
    description="Face recognition application using FaceNet models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="CogWorks",
    author_email="cogworks@example.com",
    url="https://github.com/CogWorksBWSI/face_recognition_app",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
    ],
    extras_require={
        "full": [
            "facenet_models",
            "scikit-image>=0.18.0",
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
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
)
