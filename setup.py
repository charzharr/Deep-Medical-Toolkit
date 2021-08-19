
import pathlib
from setuptools import setup, find_packages

current_dir = pathlib.Path(__file__).parent
readme = (current_dir / "README.md").read_text()
requirements = [
    'SimpleITK!=2.0.*',  # https://github.com/SimpleITK/SimpleITK/issues/1239
    'nibabel',
    'numpy>=1.15',
    'scipy',
    'scikit-image',
    'Pillow',
    'torch>=1.1',
    'torchvision'
    'torch-summary',
    'pandas',
    'psutil',
    'matplotlib',
    'ntpath',
    'wandb'
    'tqdm',
]

setup(
    name="deep-medical-toolkit",
    keywords='dmt',
    version="0.0.1",
    description=("Tools to facilitate deep learning research with a focus on "
    			 "medical imaging."),
    long_description=readme,
    long_description_content_type="text/markdown",
    # url="TBA",
    author="Charley Zhang",
    author_email="yzhang46@nd.edu",
    license="Apache License 2.0",
    classifiers=[
        "License :: Apache License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
    packages=find_packages(include=["dmt", "dmt.*"]),
    include_package_data=True,
    install_requires=requirements,
)
