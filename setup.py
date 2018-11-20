import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pypupil",
    version="0.1.9",
    author="Suyeon Choi",
    author_email="suyn.choi@gmail.com",
    description="Eye tracker (Pupil-labs) helper in Python3",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/choisuyeon/pypupil",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

    install_requires=['scipy', 'sklearn', 'numpy', 'zmq', 'msgpack'],
)
