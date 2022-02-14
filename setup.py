from setuptools import find_packages, setup

setup(
  name='momaapi',
  version='1.0',
  author='Alan Luo',
  url='https://moma.stanford.edu/',
  description='MOMA dataset API',
  python_requires='>=3.9',
  packages=find_packages(where='momaapi'),
  install_requires=[
    'distinctipy',
    'matplotlib',
    'numpy',
    'pygraphviz',
    'scipy',
    'seaborn',
    'torchvision'
  ]
)
