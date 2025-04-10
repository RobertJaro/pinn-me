import setuptools
from setuptools import setup

setup(
    name='pinn-me',
    version='0.1',
    packages=setuptools.find_packages(),
    url='https://github.com/RobertJaro/pinn-me',
    license='GNU GENERAL PUBLIC LICENSE',
    author='Robert Jarolim, Momchil Molnar',
    author_email='',
    description='Spectropolarimetric Milne-Eddington Inversions with Physics-Informed Neural Networks',
    install_requires=['torch>=1.8', 'sunpy[all]>=3.0', 'tqdm', 'wandb>=0.13',
                      'lightning==1.9.3', 'pytorch_lightning==1.9.3'],
    entry_points={
        'console_scripts': [
            'pinn-me-inversion = pme.inversion:main',
            'pinn-me-to-npz = pme.convert.pme_to_npz:main',
        ]}
)
