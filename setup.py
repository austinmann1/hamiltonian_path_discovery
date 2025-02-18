from setuptools import setup, find_packages

setup(
    name="hamiltonian_path_discovery",
    version="0.1",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy",
        "networkx",
        "pytest",
        "requests"
    ]
)
