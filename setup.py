import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deepprojection",
    version="0.1.1",
    author="Cong Wang",
    author_email="wangimagine@gmail.com",
    description="An SPI classifier",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/carbonscott/deepprojection",
    keywords = ['Single particle imaging', 'X-ray', 'Siamese network'],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
