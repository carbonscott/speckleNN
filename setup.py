import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="speckleNN",
    version="1.0.0",
    author="Cong Wang",
    author_email="wangimagine@gmail.com",
    description="An SPI speckle pattern classifier",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/carbonscott/speckleNN.git",
    keywords = ['Speckle Pattern', 'Neural network', 'X-ray single particle imaging'],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
