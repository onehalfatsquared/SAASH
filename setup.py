from setuptools import find_packages, setup

setup(
    name='SAASH',
    packages=['structure','util'],
    package_dir={"":"SAASH"},
    version='0.1.0',
    description='Self-Assembly Analysis Script for HOOMD',
    author='onehalfatsquared',
    license='MIT',
    test_suite='testing',
)