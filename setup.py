from setuptools import setup
from setuptools import find_packages

setup(
    name='categorical-kernels',
    version='0.2.0',
    packages=find_packages(),
    description='Experiments with Categorical Kernels',
    long_description=open('README.md').read(),
    url='https://github.com/Alkxzv/categorical-kernels',
    license='LICENSE',
    author='Carlos',
    install_requires=[
        "numpy >= 1.7",
        "scikit-learn >= 0.14",
        "scipy >= 0.12"
        ]
    )
