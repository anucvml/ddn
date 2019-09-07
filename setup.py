# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='ddn',
    version='1.0.0',
    description='Deep declarative networks',
    long_description=readme,
    author='Stephen Gould',
    author_email='stephen.gould@anu.edu.au',
    url='https://deepdeclarativenetworks.com',
    license=license,
    packages=['ddn']
)
