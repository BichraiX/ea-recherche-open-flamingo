from setuptools import setup, find_packages

setup(
    name='train',
    version='0.1',
    packages=find_packages(include=['defense', 'defense.*']), 
)
