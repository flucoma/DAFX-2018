## Stationary/Transient Audio Separation using Convolutional Autoencoders

This repository contains the code to reproduce the method presented in:

> Roma, G., Green, O. & Tremblay, P. A., Stationary/Transient Audio Separation using Convolutional Autoencoders. Proceedings of the 21st International Conference on Digital Audio Effects (DAFX 2018)


### Requirements:

- Python 3
- Pytorch
- untwist

### Usage:

See usage message for cae1.py, cae2.py, cae3.py. The name of the pickle file for the pre-trained models is hard-coded so change it for mono/polyphonic material. The 4x1 models correspond to cae1 and cae2, while the 4x2 correspond to cae3. Use the pre-training scripts with some audio dataset to create your own models, but note that usable parameter range may change due to random initialization.

Here are some examples:

http://www.flucoma.org/DAFX-2018/
