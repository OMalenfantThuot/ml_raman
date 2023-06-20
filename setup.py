from setuptools import setup

requirements = ["abipy==0.9.3", "h5py==3.8.0", "numpy==1.23"]

setup(
    name="ml_raman",
    description="New repository for the Raman spectra prediction of graphene from machine learning models.",
    packages=["ml_raman"],
    install_requires=requirements,
    classifiers=["Programming Language :: Python :: 3.10"],
)
