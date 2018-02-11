# VIND (Variational Inference for Nonlinear Dynamics)


This code represents a reference implementation for the paper "[Variational Inference for Nonlinear Dynamics](https://github.com/dhernandd/vind/blob/master/paper/nips_workshop.pdf)", accepted for the Time Series Workshop at NIPS 2017. It represents a sequential variational autoencoder that is able to infer nonlinear dynamics in the latent space. The training algorithm makes use of a novel, two-step technique for optimization based on the [Fixed Point Iteration](https://en.wikipedia.org/wiki/Fixed-point_iteration) method for finding fixed points of iterative equations.

# Installation

The code is writeen in Python 2.7. You will need the bleeding edge versions of the following packages:

- Theano
- Lasagne

In addition, up-to-date versions of numpy, scipy and matplotlib are expected.

# Running the code

In order to run the runner script, execute

```sh
$ python runner.py
```

This should run as is, and proceed to attempt to fit the example dataset provided in the data directory. If not present, it will create a directory in your local machine to store the results. To fit any other dataset, you may change the relevant options inside the runner file, as well as a host of others that may have an impact on the quality of the fit. 
