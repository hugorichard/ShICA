from setuptools import setup

setup(
    name="shica",
    version="0.0.0",
    description="Shared ICA",
    license="MIT",
    keywords="ICA",
    install_requires=[
        "numpy>=1.12",
        "scikit-learn>=0.23",
        "scipy>=0.18.0",
        "matplotlib>=2.0.0",
        "qndiag",
        "mvlearn",
    ],
)
