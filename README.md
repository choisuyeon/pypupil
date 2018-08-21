# pypupil (beta)
###### Eye tracker (Pupil-labs) helper in Python3
- Calibration without World Camera
- Record Pupil data and conversion to .mat file
- Synchronization of both eye tracking data

## API docs
[Pypupil docs](https://choisuyeon.github.io/docs-pypupil)

## Getting Started
You should read the official docs from Pupil-labs  
: how to setup pupil-labs eye tracker [Pupil labs docuentation](https://docs.pupil-labs.com/#developer-setup)


### Prerequisites
scipy, sklearn, numpy, zmq, msgpack, matlab, matplotlib
You can install all of those libraries via pip

such as
```
pip install scipy
```


## How to use
1. You can use Command Line Interface
```
python pypupil_main.py
```
2. or use the module with given [API](#)

## Built With

* [SciPy](https://www.scipy.org/) - Python-based ecosystem of open-source software for mathematics, science, and engineering
* [scikit-learn](https://maven.apache.org/) - Simple and efficient tools for data mining and data analysis
* [numpy](https://scikit-learn.org/) - NumPy is the fundamental package for scientific computing with Python
* [ZeroMQ](https://zeromq.org/) - A high-performance asynchronous messaging library
* [MessagePack](https://msgpack.org/) - MessagePack is an efficient binary serialization format
* [python3-tk](#) - TBU
* [matplotlib](#) - TBU
* [MATLAB](#) - TBU

## Authors

* **Suyeon Choi** - *OEQELAB, SNU* - [choisuyeon](https://github.com/choisuyeon)
###### Contact : 0310csy@hanmail.net*

## References
- [Pupil-labs docs](https://docs.pupil-labs.com/#developer-docs)
- [Pupil-labs docs synchronization](https://docs.pupil-labs.com/#multi-camera-synchronization)
- [Affine Transformer By Jarno Elonen](https://elonen.iki.fi/code/misc-notes/affine-fit/)
