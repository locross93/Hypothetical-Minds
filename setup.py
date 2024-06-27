from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="Hypothetical-Minds",
    version="0.1",
    packages=find_packages(),
    author='Logan',
    author_email='locross93@gmail.com',
    url='https://github.com/locross93/Hypothetical-Minds',
    license='LICENSE',
    install_requires=required,
    dependency_links=[
        'file:./meltingpot#egg=meltingpot'
    ],
    python_requires='>=3.8',
)
