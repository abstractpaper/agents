import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="agents",
    version="0.0.1",
    author="Aziz Alfoudari",
    author_email="aziz.alfoudari@gmail.com",
    description="Reinforcement Learning agents implemented in pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/abstractpaper/agents",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)