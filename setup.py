from setuptools import setup, find_packages

setup(
    name='AbFlex',
    version='0.0.1',
    author='Fabian C. Spoendlin',
    author_email='fabian.spoendlin@stats.ox.ac.uk',
    description='Prediction of antibody CDR and protein loop flexibility',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/fspoendlin/AbFlex',
    packages=find_packages('AbFlex', 'AbFlex.*', 'AbFlex_seq', 'AbFlex_seq.*'),
)