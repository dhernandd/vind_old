# VIND (Variational Inference for Nonlinear Dynamics)


This code represents a reference implementation for the paper:

D. Hernandez, L. Paninski and J. Cunningham; *Variational Inference for Nonlinear Dynamics*, accepted for the Time Series Workshop at NIPS 2017.

# Installation

You will need the bleeding edge versions of the following packages:

- Theano
- Lasagne

In addition, up-to-date versions of numpy, scipy and matplotlib are expected.

# Running the code

The runner file should run as is, and proceed to attempt to fit the data provided in the data directory. If not present, it will create a directory in your local machine to store the results. To fit any other data, you may change the relevant options inside the runner file, as well as a host of others that may have an impact on the quality of the fit. Options are partialy documented at the moment.
