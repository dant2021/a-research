from setuptools import setup, find_packages

setup(
    name="emotional-speech-pipeline",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "numpy>=1.21.0",
        "openai-whisper>=20231117",
        "kokoro>=0.1.0",
    ],
    author="Antoine Descamps",
    description="A neural pipeline for preserving emotional characteristics in speech-to-speech conversion",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/dant2021/a-research/emotional-speech-pipeline",
    classifiers=[
        "Development Status :: Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.10",
) 