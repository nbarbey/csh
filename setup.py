#!/usr/bin/env python
from setuptools import Extension, setup
setup(name='Csh',
      version='1.0',
      description='Compressive Sensing for Herschel',
      author='Nicolas Barbey',
      author_email='nicolas.barbey@cea.fr',
      install_requires = ['numpy>=1.3.0', 'fht', ],
      packages=['csh'],
      scripts=['csh/pacs_compression.py'],
      )
