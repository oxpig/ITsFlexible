from setuptools import setup, find_packages

setup(
    name='ITsFlexible',
    version='1.0.0',
    author='Fabian C. Spoendlin',
    author_email='fabian.spoendlin@stats.ox.ac.uk',
    description='Prediction of antibody CDR and protein loop flexibility',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/fspoendlin/ITsFlexible',
    packages=find_packages(include=['ITsFlexible', 'ITsFlexible.*']),
    include_package_data=True,
    package_data={
        'ITsFlexible': ['trained_model/**'],
    },
    install_requires=[
        'biopandas==0.4.1',
        'biopython==1.83',
        'fastparquet==2024.2.0',
        'numpy==1.26.4',
        'pandas==2.2.1',
        'pyarrow==14.0.2',
        'lightning==2.3.3',
        'scikit-learn==1.5.0',
        'tqdm==4.66.4',
        'wandb==0.17.0',
    ],
)
