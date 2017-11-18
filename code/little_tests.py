import numpy as np

import theano
import theano.tensor as T
import lasagne
from lasagne.nonlinearities import softmax, linear


# ================================================ 

# Demonstration of how to compute the missing term in the posterior recurrent
# equation

xsamp = T.matrix('x')
specify_shape = T.specify_shape(xsamp, (3, 2))
Nsamps, _ = xsamp.shape[0], xsamp.shape[1]
xDim = 2

# Define the NN.
NNEvolve = lasagne.layers.InputLayer((None, xDim), name='IL')
NNEvolve = lasagne.layers.DenseLayer(NNEvolve, 30, nonlinearity=softmax, 
                                     W=lasagne.init.Orthogonal(), name='_HL1')
NNEvolve = lasagne.layers.DenseLayer(NNEvolve, xDim**2, nonlinearity=linear, 
                                          W=lasagne.init.Uniform(0.9), name='_OL')
B = lasagne.layers.get_output(NNEvolve, xsamp)
B = T.sum(xsamp**2)

# samp = np.random.rand(3, 2)
# B_func = theano.function([xsamp], B)
# print 'B:', B_func(samp)

x0 = xsamp[1]
test = T.grad(B, x0)
# print 'test:', test.eval({xsamp : samp})

test2 = T.grad(B, xsamp)
# print 'test:', test2.eval({xsamp : samp})

assert False

# Flatten the output. This is needed in theano for gradient computation.
flatB = B.flatten()
# Since x is a matrix, the gradient of flatB w.r.t. it is a tensor3. In the 0th
# dimension, we have the total number of entries in flatB. Dim 1 is the number
# of samps and dim 2 is xDim
flat_gradB_Nd2xNxd, _ = theano.scan(lambda i, b, x : T.grad(b[i], x[i:i+1]), 
                                    sequences=T.arange(flatB.shape[0]),
                                    non_sequences=[flatB, xsamp])

# Each entry in flatB corresponds to a sample and depends only on the x of that
# sample. However, the grad above takes the derivatives of all Bs w.r.t. ALL
# samps. Which means the tensor above is full of zeros. I reshape first to put
# Nsamps in the leading dim...
flat_gradB2 = flat_gradB_Nd2xNxd.reshape([Nsamps, xDim**2, Nsamps, xDim])
# ... and now slice the ii elements which are the only nonzero ones. Then
# reshape each samp to xDim x xDim x xDim, the first two dimensions
# corresponding to the entry in B, the last one indicating which x the
# derivative was taken. Finally reshuffle to put the entries of B last.
flat_gradB3_Nxdxdxd, _ = theano.scan(lambda i, X : X[i,:,i,:].reshape([xDim, xDim, xDim]).dimshuffle(2, 0, 1), 
                            sequences=T.arange(Nsamps),
                            non_sequences=[flat_gradB2])
# Now, take the batched dot.
dotted_Nxd = T.batched_dot(xsamp, T.batched_dot(flat_gradB3_Nxdxdxd, xsamp))

flatgrad = theano.function([xsamp], flat_gradB_Nd2xNxd)
J = theano.function([xsamp], flat_gradB3_Nxdxdxd)
J2 = theano.function([xsamp], dotted_Nxd)

N = 3
x = np.random.rand(N, xDim)
# J1 = J0.reshape([2, xDim**2, 2, xDim])
print 'x:', x
print 'flatgrad:', flatgrad(x)
print 'J:', J(x)
print 'J2:', J2(x)


# J2 = [J1[i,:,i,:].T for i in range(2)]
# print 'J2:', J2
# 
# # len(J3) is the number of samps. 
# # Each entry in J3 is the numpy array formed by $dB_{ij}/dx_k$. i, j, k are in {0,1} and
# # the shape is in the 'kij' order (that is, 0th index stands for the coordinate x_k etc.)  
# J3 = [np.reshape(J2[i], (xDim, xDim, xDim)) for i in range(len(J2))]
# print 'J3:', J3

