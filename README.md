# Operationally meaningful representations of physical systems in neural networks

Code for: H. Poulsen Nautrup, T. Metger, R. Iten, S. Jerbi, L.M. Trenkwalder, H.Wilming, H.J. Briegel, and R. Renner. "Operationally meaningful representations of physical systems in neural networks" (2020).

This repository contains the trained [Tensorflow](https://www.tensorflow.org) models used in section 5 of the paper as well as code to load, train and analyze them. The code for the example using reinforcement learning (section 6 in the paper) can be found [here](https://github.com/HendrikPN/reinforced_scinet).

Requires:

- ``Python 3.6.7``
- ``numpy 1.16.2``
- ``scipy 1.2.1``
- ``matplotlib 3.0.3``
- ``tensorflow 1.13.1``
- ``tensorboard 1.13.1``
- ``tqdm 4.31.1``
- ``jupyter 1.0.0``

To use the code:

1. Clone the repository.
2. Add the cloned directory ``communicating_scinet`` to your python path. See [here](https://stackoverflow.com/questions/10738919/how-do-i-add-a-path-to-pythonpath-in-virtualenv) for instructions for doing this in a virtual environment. Without a virtual environment, see [here](https://stackoverflow.com/questions/3402168/permanently-add-a-directory-to-pythonpath).

Generated data files are stored in the ``data`` directory. Saved models are stored in the ``tf_save`` directory. Tensorboard logs are stored in the ``tf_log`` directory.

Some documentation is available in the code. For further questions, please contact us directly.

