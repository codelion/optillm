from setuptools import setup, find_packages

setup(
    name="optillm",
    version="0.0.10",
    packages=find_packages(),
    py_modules=['optillm'],
    package_data={
        'optillm': ['plugins/*.py'],  # Include plugin files
    },
    include_package_data=True,  # This is important
    install_requires=[
        "numpy",
        "networkx",
        "openai",
        "z3-solver",
        "aiohttp",
        "flask",
        "torch",
        "transformers",
        "azure-identity",
        "tiktoken",
        "scikit-learn",
        "litellm",
        "requests",
        "beautifulsoup4",
        "lxml",
        "presidio_analyzer",
        "presidio_anonymizer",
        "nbconvert",
        "nbformat",
        "ipython",
        "ipykernel",
    ],
    entry_points={
        'console_scripts': [
            'optillm=optillm:main',  # Points directly to the main function in optillm.py
        ],
    },
    author="codelion",
    author_email="codelion@okyasoft.com",
    description="An optimizing inference proxy for LLMs.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/codelion/optillm",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
