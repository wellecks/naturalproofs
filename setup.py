from setuptools import setup
import sys

if sys.version_info < (3,):
    sys.exit('Sorry, Python3 is required.')

with open('requirements.txt') as f:
    reqs = f.read()

setup(
    name='naturalproofs',
    version='1.0.0',
    description='naturalproofs',
    packages=['naturalproofs'],
    install_requires=reqs.strip().split('\n'),
    include_package_data=True,
)