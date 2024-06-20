from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()


setup(
    name='llm_plan',  # Replace with your package name
    version='0.1.0',  # Replace with your package version
    description='LLM plan for multi-agent systems',  # Provide a short description
    author='Violet',  # Replace with your name
    author_email='ziyxiang@stanford.edu',  # Replace with your email
    url='https://github.com/violetxi/llmv_marl',  # Replace with your repository URL
    license='LICENSE',  # Specify the license
    packages=find_packages(),
    install_requires=required,
    entry_points={
        'console_scripts': [
            'llm_plan=llm_plan.main:main', # Adjust if your entry point differs
        ],
    },
    include_package_data=True,
    package_data={
        # Include any package data here
    },
    classifiers=[
        # Classifiers help users find your project
        # For a full list, see https://pypi.org/classifiers/
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',  # Specify your Python version requirement
)
