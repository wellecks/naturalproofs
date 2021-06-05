from setuptools import setup

with open('requirements.txt') as f:
    reqs = f.read()

setup(
    name='naturalproofs',
    version='2.0.0',
    description='naturalproofs',
    packages=['naturalproofs'],
    install_requires=reqs.strip().split('\n'),
    include_package_data=True,
)