"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from setuptools import find_packages, setup

setup(
    name="omdata",
    version="0.1.0",
    description="Code for generating OMOL input configurations",
    url="http://github.com/Open-Catalyst-Project/om-data",
    packages=find_packages(),
    install_requires=[
        "ase>=3.24.0",
        "quacc>=0.12.1",
        "sella>=2.3.5",
        "numpy<2"
    ],
    include_package_data=True,
)
