from setuptools import setup, find_packages

setup(
    name="data_assimilation",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "jupyter",
        "tqdm",
        "ipywidgets" # Add more dependencies if needed
    ],
)