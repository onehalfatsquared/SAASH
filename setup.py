from setuptools import find_packages, setup

setup(
    name='SAASH',
    packages=find_packages(include=["SAASH","SAASH.structure","SAASH.util"]),
    version='0.2.0',
    description='Self-Assembly Analysis Script for HOOMD',
    author='onehalfatsquared',
    license='MIT',
    test_suite='testing',
    install_requires=['numpy','pandas','gsd','scipy']
)
