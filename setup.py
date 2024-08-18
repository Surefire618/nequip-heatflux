from setuptools import setup, find_packages

setup(
    name="nequip-heatflux",
    version="0.0.1",
    description="Heat flux calculator for NequIP",
    download_url="https://github.com/Surefire618/nequip-heatflux",
    author="Shuo Zhao",
    python_requires=">=3.7",
    packages=find_packages(include=["nequip_calculator", "nequip_calculator.*"]),
    install_requires=[
        "numpy",
        "ase",
        "typing_extensions;python_version<'3.8'",  # backport of Final
        "importlib_metadata;python_version<'3.10'",  # backport of importlib
        "torch-runstats>=0.2.0",
        "torch-ema>=0.3.0",
    ],
    zip_safe=True,
)
