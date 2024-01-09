#!/usr/bin/env python
from setuptools import setup, find_packages
import os
from pathlib import Path

here = Path(__file__).parent.absolute()

# Get the long description from the README file
with open(here.joinpath('README.md'), encoding='utf-8') as f:
    long_description = f.read()
    setup(    
        name='warming_levels',    
        version='0.1.0',    
        description='Woodwell Climate Research Center | Functions for working with CMIP6 data at warming levels',    
        long_description=long_description,    
        long_description_content_type="text/markdown",    
        url='https://github.com/abbylute/Warming_levels',    
        author='Woodwell Climate Research Center',    
        author_email='alute@woodwellclimate.org',    
        license='TBD',    
        packages=find_packages(where = 'src'),
        package_dir={"": "src"})
