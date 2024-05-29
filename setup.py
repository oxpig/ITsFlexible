from setuptools import setup, find_packages

setup(
    name='AbFlex',
    version='0.0.1',
    author='Fabian C. Spoendlin',
    description='EGNN for classifying the flexibility of protein loops and antibody CDRs',
    packages=find_packages('src', 'src.*'),
    install_requires=[
    ],
)