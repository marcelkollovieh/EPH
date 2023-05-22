from setuptools import setup, find_packages

setup(
    name="eph_clustering",
    description="Expected Probabilistic Hierarchies",
    long_description="Expected Probabilistic Hierarchies",
    version="0.0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch_geometric==2.0.4",
        "torch_sparse==0.6.14",
        "torch_scatter==2.0.8",
        "networkx==2.7.1",
        "numpy~=1.21",
        "HeapDict==1.0.1",
        "seml==0.3.5",
        "setuptools",
        "sacred",
        "jupyter",
    ],
)
