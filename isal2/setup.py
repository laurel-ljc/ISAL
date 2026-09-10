"""Map the existing top-level layout to the isal2 package."""
from setuptools import find_packages, setup

setup(packages=["isal2"] + ["isal2." + p for p in find_packages(".")],
      package_dir={"isal2": "."}, include_package_data=True)
