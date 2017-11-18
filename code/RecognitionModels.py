import numpy as np 

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
srng = RandomStreams(seed=1)

import lasagne
from lasagne.nonlinearities import linear, softplus 

from LatEvModels import LocallyLinearEvolution


from utils import blk_tridiag_chol, blk_chol_inv


class RecognitionModelTS(object):
    """
    This superclass models a variational posterior for a time series {X_t} |
    {Y_t} (X_t given Y_t for all t)
    
    X_t stands for a 'low' dimensional hidden time series.
    Y_t stands for a 'high' dimensional time series of observations.
    Q(X_t|Y_t) is the proposal posterior distribution.

    NOTE: A Recognition Model is a ...
    """
    def __init__(self, RecPars, yDim, Y=None):
        """
        """
        for key in RecPars.keys():
            setattr(self, key, RecPars[key])
        
        self.yDim = yDim
        self.Y = T.tensor3('Y') if Y is None else Y 
        
    def get_yDim(self):
        return self.yDim
    
    
    def sample_X(self):
        raise Exception('\nPlease define me. This is an abstract method.')
    
    
    def compute_Entropy(self):
        raise Exception('\nPlease define me. This is an abstract method.')



class SmoothingTimeSeries(RecognitionModelTS):
    """
    """
    def __init__(self, RecPars, yDim, xDim, Y=None, X=None):
        """
        Input : 'y' theano.tensor.var.TensorVariable (n_input)
                Observation matrix based on which we produce q(x)
            - RecognitionParams : (dictionary)
                Dictionary of timeseries-specific parameters. Contents:
                     * A -
                     * NN paramers...
                     * others... TODO
            - xDim, yDim, zDim : (integers) dimension of
                latent space (x) and observation (y)
        """
        RecognitionModelTS.__init__(self, RecPars, yDim, Y)
        
        self.Nsamps = Nsamps = self.Y.shape[0]
        self.Tbins = Tbins = self.Y.shape[1]
        self.xDim = xDim
        
        recnodes = 60
        if not hasattr(self, 'NNMuX'):
            NNMuX = lasagne.layers.InputLayer((None, None, yDim), name='MuX_IL')
            NNMuX = lasagne.layers.DenseLayer(NNMuX, recnodes, nonlinearity=softplus, W=lasagne.init.Normal(0.05), num_leading_axes=2, name='MuX_HL1')
            NNMuX = lasagne.layers.DenseLayer(NNMuX, recnodes, nonlinearity=softplus, W=lasagne.init.Normal(0.05), num_leading_axes=2, name='MuX_HL2')
            self.NNMuX = lasagne.layers.DenseLayer(NNMuX, xDim, nonlinearity=linear, W=lasagne.init.Orthogonal(), num_leading_axes=2, name='MuX_OL')
        self.MuX = lasagne.layers.get_output(self.NNMuX, inputs=self.Y)
        Mus = T.reshape(self.MuX, [Nsamps*Tbins, xDim])
        
        if not hasattr(self, 'NNLambdaX'):
            NNLambdaX = lasagne.layers.InputLayer((None, None, yDim), name='LX_IL')
            NNLambdaX = lasagne.layers.DenseLayer(NNLambdaX, recnodes, nonlinearity=softplus, W=lasagne.init.Normal(0.05), num_leading_axes=2, name='LX_HL1')
            NNLambdaX = lasagne.layers.DenseLayer(NNLambdaX, recnodes, nonlinearity=softplus, W=lasagne.init.Normal(0.05), num_leading_axes=2, name='LX_HL1')
            self.NNLambdaX = lasagne.layers.DenseLayer(NNLambdaX, xDim**2, nonlinearity=linear, W=lasagne.init.Orthogonal(), num_leading_axes=2, name='MuX_OL')
        self.NNLambdaX_out = lasagne.layers.get_output(self.NNLambdaX, inputs=self.Y)
        self.LambdaChol = T.reshape(self.NNLambdaX_out, [Nsamps, Tbins, xDim, xDim])
        LambdaChols = T.reshape(self.LambdaChol, [Nsamps*Tbins, xDim, xDim])

        Lambdas = T.batched_dot(LambdaChols, LambdaChols.dimshuffle(0,2,1))
        self.Lambda = T.reshape(Lambdas, [Nsamps, Tbins, xDim, xDim])
        
        self.LambdaMu = T.batched_dot(Lambdas, Mus)
        self.LambdaMu = T.reshape(self.LambdaMu, [Nsamps, Tbins, xDim])   # Posterior Mean
                 

    def eval_Mu(self, Ydata):
        """
        """
        return self.MuX.eval({self.Y : Ydata})
        
    
    def eval_LambdaChol(self, Ydata):
        """
        """
        return self.LambdaChol.eval({self.Y : Ydata})
    
    
    def eval_Lambda(self, Ydata):
        """
        """
        return self.Lambda.eval({self.Y : Ydata})


#========================================================================

class SmoothingLDSTimeSeries(SmoothingTimeSeries):
    """
    A "smoothing" recognition model for a time series based on a hidden LDS.

    x ~ N( mu(y), sigma(y) )

    """
    def __init__(self, RecPars, SmthLDSTSPars, yDim, xDim, GM=None, Y=None, X=None):
        """     
        Input : 'y' theano.tensor.var.TensorVariable (n_input)
                Observation matrix based on which we produce q(x)
            - RecognitionParams : (dictionary)
                Dictionary of timeseries-specific parameters. Contents:
                     * A -
                     * NN paramers...
                     * others... TODO
            - xDim, yDim, zDim : (integers) dimension of
                latent space (x) and observation (y)
        """
        
        SmoothingTimeSeries.__init__(self, RecPars, SmthLDSTSPars, yDim, xDim, GM=GM, Y=Y, X=X)
        
        self._initialize_posterior_distribution(RecPars)


    def _initialize_posterior_distribution(self, RecognitionParams):
        """
        """
        self.A = self.gm.get_A()
        self.QInv = self.gm.get_QInv()
        self.Q0Inv = self.gm.get_Q0Inv()
 
        ##### Put together the total precision matrix ######
        AQInvA = T.dot(T.dot(self.A.T, self.QInv), self.A)
        AQInvARep = T.tile(AQInvA + self.QInv, [self.Tbins-2, 1, 1])
        AQInvRep = T.tile(-T.dot(self.A.T, self.QInv), [self.Tbins-1, 1, 1]) # off-diagonal blocks (upper triangle). Replicate a bunch of times
        AQInvARepPlusQ = T.concatenate([T.shape_padleft(self.Q0Inv + AQInvA), AQInvARep, T.shape_padleft(self.QInv)])
        self.AQInvARepPlusQ = T.tile(AQInvARepPlusQ, [self.Nsamps, 1, 1, 1])
        
        # This is the inverse covariance matrix: diagonal (AA) and off-diagonal (BB) blocks.
        self.AA = self.Lambda + self.AQInvARepPlusQ
        self.BB = T.tile(AQInvRep, [self.Nsamps, 1, 1, 1])

        # compute Cholesky decomposition
        self.TheChol, _ = theano.scan(lambda A, B : blk_tridiag_chol(A, B), sequences = (self.AA, self.BB))
        def postX_from_chol(tc1, tc2, lm):
            return blk_chol_inv(tc1, tc2, blk_chol_inv(tc1, tc2, lm))
        self.postX, _ = theano.scan(lambda tc1, tc2, lm : postX_from_chol(tc1, tc2, lm), sequences=(self.TheChol[0], self.TheChol[1], self.LambdaMu))

        # Determinant of the Cholesky factor is the product of the diagonal elements of the block-diagonal.
        def comp_log_det(L):
            return T.log(T.diag(L)).sum()
        self.LnDeterminant = -2.0*theano.scan(fn=comp_log_det, 
                    sequences=T.reshape(self.TheChol[0], [self.Nsamps*self.Tbins, self.xDim, self.xDim]))[0].sum()
    
    
    def eval_TheChol(self, Ydata):
        """
        """
        return self.TheChol[0].eval({self.Y : Ydata}), self.TheChol[1].eval({self.Y : Ydata})
    
    
    def eval_postX(self, Ydata):
        """
        """
        return self.postX.eval({self.Y : Ydata})
    

    def symbsample_X(self, Y=None):
        Y = self.Y if Y is None else Y
        Nsamps, Tbins = Y.shape[0], Y.shape[1]
        
        TheChol = theano.clone(self.TheChol, replace={self.Y : Y})
        postX = theano.clone(self.postX, replace={self.Y : Y})
        normSamps = srng.normal([Nsamps, Tbins, self.xDim])
        noise, _ = theano.scan(lambda tc1, tc2, ns : blk_chol_inv(tc1, tc2, ns, lower=False, transpose=True), 
                            sequences=(TheChol[0], TheChol[1], normSamps))
        return postX + noise


    def sample_X(self, Ydata):
        return self.symbsample_X().eval({self.Y : Ydata})


    def compute_Entropy(self, Y=None):
        Y = self.Y if Y is None else Y
        Nsamps, Tbins = Y.shape[0], Y.shape[1]
        
        LnDeterminant = theano.clone(self.LnDeterminant, replace={self.Y : Y})
        Entropy = 0.5*LnDeterminant + 0.5*Nsamps*Tbins*(1 + np.log(2*np.pi))
        return Entropy


    def eval_Entropy(self, Ydata):
        Nsamps = Ydata.shape[0]
        return self.compute_Entropy().eval({self.Y : Ydata})/Nsamps


    def get_Weights(self):
        layers_Mu = lasagne.layers.get_all_layers(self.NN_Mu)
        layers_Lambda = lasagne.layers.get_all_layers(self.NN_Lambda)
        return [l.W for l in layers_Mu[1:]] + [l.W for l in layers_Lambda[1:]]
 

    def get_Params(self):
        return lasagne.layers.get_all_params(self.NNMuX) + lasagne.layers.get_all_params(self.NNLambdaX)


#==================================================================

def covariance_time_series_grad(time_series_NxTxdxd, Y, X, xDim, remove_from_tail=0):
        """
        TODO: Write docstring
        
        Args:
            time_series_NxTxdxd: N samples of a time series of square matrices 
                    of dimension xDim. Each of these matrices (N*T of them) depends 
                    on one of the N*T independent variables in self.X 
        Output:
            time_series_grad_NTxdxdxd: The NT gradients with respect to the independent
                    variables. Dim 1 is the gradient dimension while dims 2 and 3 
                    stand for the dimensions of the original square matrix.
        """        
        N, Tb = time_series_NxTxdxd.shape[0], time_series_NxTxdxd.shape[1]
        time_series_flat_NTdd = time_series_NxTxdxd.flatten()
        
        def lambda1(i, b, x): return T.grad(b[i], x)
        time_series_grad_NTddxNxTxd, _ = theano.scan(lambda1, 
                                            sequences=T.arange(time_series_flat_NTdd.shape[0]),
                                            non_sequences=[time_series_flat_NTdd, X])
#         print 'T1:', time_series_grad_NTddxNxTxd.eval({X : sampleX, Y : sampleY})[0,0,0,:]
        
        if remove_from_tail:
            time_series_grad_NTddxNxTxd = time_series_grad_NTddxNxTxd[:,:,:-remove_from_tail,:]
#             print 'grad:', time_series_grad_NTddxNxTxd.eval({self.X : sampleX, self.Y : sampleY})
        time_series_grad_NTxddxNTxd = time_series_grad_NTddxNxTxd.reshape([N*Tb,
                                                            xDim**2, N*Tb, xDim])
        def lambda2(i, X): return X[i,:,i,:].reshape([xDim, xDim, xDim]).dimshuffle(2, 0, 1)
        time_series_grad_NTxdxdxd, _ = theano.scan(lambda2, 
                        sequences=T.arange(N*Tb),
                        non_sequences=[time_series_grad_NTxddxNTxd])
        
        return time_series_grad_NTxdxdxd


class SmoothingNLDSTimeSeries(SmoothingTimeSeries):
    """
    A "smoothing" recognition model for a time series based on a hidden LDS.

    x ~ N( mu(y), sigma(y) )

    """
    def __init__(self, RecPars, yDim, xDim, Y=None, X=None, lat_ev_model=None, LATCLASS=LocallyLinearEvolution):
        """     
        Input : 'y' theano.tensor.var.TensorVariable (n_input)
                Observation matrix based on which we produce q(x)
            - RecognitionParams : (dictionary)
                Dictionary of timeseries-specific parameters. Contents:
                     * A -
                     * NN paramers...
                     * others... TODO
            - xDim, yDim, zDim : (integers) dimension of
                latent space (x) and observation (y)
        """
        if lat_ev_model is None:
            self.common_lat = False
            self.X = X = T.tensor3('X') if X is None else X
            self.lat_ev_model = lat_ev_model = LATCLASS({}, xDim, X, nnname='RecEv')
        else:
            self.common_lat = True
            self.lat_ev_model = lat_ev_model
            self.X = X = lat_ev_model.get_X() if X is None else X

        SmoothingTimeSeries.__init__(self, RecPars, yDim, xDim, Y=Y, X=X)

        self._initialize_posterior_distribution()
        
    
    def _initialize_posterior_distribution(self):
        """
        Computes the posterior mean and variance.
        
        The proposal distribution q is Generalized Gaussian in the full set of
        hidden variables:
        
        q(z) ~ exp{ (z-Mu)^T*Lambda*(z-Mu) + z^T*Omega(z)*z }
        
        where Omega is tridiagonal and depends on z. 
        """
        
        self.QInv = self.lat_ev_model.get_QInv()
        self.Q0Inv = self.lat_ev_model.get_Q0Inv()
        self.totalA = T.reshape(self.lat_ev_model.get_totalA(), [self.Nsamps*(self.Tbins-1), self.xDim, self.xDim])   # implicit symbolic dependence on self.gen_model.X
        self.QQInv = T.tile(self.QInv, [self.Nsamps*(self.Tbins-1), 1, 1])
        
        # Computes the block diagonal matrix:
        #     Qt^-1 = diag{Q0^-1, Q^-1, ..., Q^-1}
        QInv = T.tile(self.QInv, [self.Tbins-2, 1, 1])
        Q0Inv = T.reshape(self.Q0Inv, [1, self.xDim, self.xDim])
        Q0QInv = T.concatenate([Q0Inv, QInv], axis=0)
        self.QInvTot = T.tile(Q0QInv, [self.Nsamps, 1, 1])

        # The diagonal blocks of Omega(z) up to T-1:
        #     Omega(z)_ii = A(z)^T*Qq^{-1}*A(z) + Qt^{-1},     for i in {1,...,T-1 }
        self.AQInvA = T.batched_dot(self.totalA.dimshuffle(0, 2, 1), 
                                    T.batched_dot(self.QQInv, self.totalA)) + self.QInvTot
        self.AQInvA = self.AQInvA.reshape([self.Nsamps, self.Tbins-1, self.xDim, self.xDim])
        
        # The off-diagonal blocks of Omega(z):
        #     Omega(z)_{i,i+1} = A(z)^T*Q^-1,     for i in {1,..., T-2} 
        self.AQInv = -T.batched_dot(self.totalA.dimshuffle(0, 2, 1), self.QQInv)
        self.AQInv = self.AQInv.reshape([self.Nsamps, self.Tbins-1, self.xDim, self.xDim])
        
        # Tile in the last block Omega_TT. 
        # This one does not depend on A. There is no latent evolution beyond T.
        QInvRep = T.tile(self.QInv, [self.Nsamps, 1, 1, 1])
        self.AQInvAPlusQ = T.concatenate([self.AQInvA, QInvRep], axis=1)
        
        # Add in the covariance coming from the observations
        self.AA = self.Lambda + self.AQInvAPlusQ    # NxTxdxd
        self.BB = self.AQInv    # NxT-1xdxd
        
        Xflat = self.X.reshape([self.Nsamps*self.Tbins, self.xDim])
        Xflat01 = self.X[:,:-1,:].reshape([self.Nsamps*(self.Tbins-1), self.xDim])
        Xflat10 = self.X[:,1:,:].reshape([self.Nsamps*(self.Tbins-1), self.xDim])
        
        # Computation of the variational posterior mean
        self.TheChol, _ = theano.scan(lambda A, B : blk_tridiag_chol(A, B),
                                       sequences = (self.AA, self.BB))
        def postX_from_chol(tc1, tc2, lm):
            """
            postX = (Lambda1 + Lambda2)^{-1}.Lambda1.Mu
            """
            return blk_chol_inv(tc1, tc2, blk_chol_inv(tc1, tc2, lm), lower=False, transpose=True)
        self.postX, _ = theano.scan(lambda tc1, tc2, lm : postX_from_chol(tc1, tc2, lm), 
                                    sequences=(self.TheChol[0], self.TheChol[1], self.LambdaMu))
        
        # Log of the determinant for computing the Entropy.
        def comp_log_det(L):
            return T.log(T.diag(L)).sum()
        self.LnDeterminant = -2.0*theano.scan(fn=comp_log_det, 
                        sequences=T.reshape(self.TheChol[0], 
                                            [self.Nsamps*self.Tbins, self.xDim, self.xDim]))[0].sum()
        
        #=== PROCEED TO THE METHODS OF THE CLASS FOR NOW IF YOU ARE A USER ===#
        
        def compute_grad_term():
            """
            TODO: This implementatio is way too slow. THIS IS THE MOST IMPORTANT
            THING TO BE DONE
            """
            self.grad_AA_NTxdxdxd = covariance_time_series_grad(self.AA, self.Y, self.X, self.xDim)
            self.grad_BB_NTm1xdxdxd = covariance_time_series_grad(self.BB, self.Y, self.X, 
                                                              self.xDim, remove_from_tail=1)
#             gradA_func = theano.function([self.X, self.Y], self.grad_AA_NTxdxdxd)
#             assert False
        
            test_AA = T.batched_tensordot(Xflat, 
                    T.batched_tensordot(self.grad_AA_NTxdxdxd, Xflat, axes=[[3],[1]]),
                    axes=[[1],[2]]).reshape([self.Nsamps, self.Tbins, self.xDim])
    #         test_AA_func = theano.function([self.X, self.Y], test_AA)
            test_BB = T.batched_tensordot(Xflat01, 
                    T.batched_tensordot(self.grad_BB_NTm1xdxdxd, Xflat10, axes=[[3],[1]]),
                    axes=[[1],[2]])
            test_BB = T.concatenate([test_BB.reshape([self.Nsamps, self.Tbins-1, self.xDim]), 
                                     T.zeros([self.Nsamps, 1, self.xDim])], axis=1)
            final_term = -0.5*test_AA - test_BB
            return final_term
    #         test_BB = T.batched_dot(a, b)
    #         print 'Doing the gradients'
    #         grad_AA = covariance_time_series_grad(self.AA)
    #         grad_BB_func = theano.function([self.X, self.Y], self.grad_BB_NxTm1xdxdxd)
    
    #         test_BB_func = theano.function([self.X, self.Y], test_BB)
    #         sampleY = np.random.rand(2, 3, 10)
    #         sampleX = np.random.rand(2, 3, 2)
    #         print 'Finished gradients'
    #         print 'GradBB:', grad_BB_func(sampleX, sampleY)
    #         sampleY = np.random.rand(2, 3, 10)
    #         sampleX = np.random.rand(2, 3, 2)        
    #         print 'Eval gradients:', test_AA_func(sampleX, sampleY)
    #         print 'Eval gradients:', test_BB_func(sampleX, sampleY)
    #         assert False
        
        # Compute Cholesky decomposition for all samples.
        # A is Txdxd, B is T-1xdxd
#         self.TrueSol = self.LambdaMu + final_term
#         
# #         sampleY = np.random.rand(2, 3, 10)
# #         sampleX = np.random.rand(2, 3, 2)        
# #         print 'final', test_AA.eval({self.Y : sampleY, self.X : sampleX})
#         self.Xsol, _ =  theano.scan(lambda tc1, tc2, lm : postX_from_chol(tc1, tc2, lm), 
#                                     sequences=(self.TheChol[0], self.TheChol[1], self.TrueSol))
        
        # Determinant of the Cholesky factor is the product of the diagonal elements of the block-diagonal.

    
    def __getstate__(self):
        """
        Terrible hack. Needed it because this fucker was not fucking pickling.
        Max recursion limit exceeded, blah blah blah... Fucker.
        """
        return {key : value for key, value in self.__dict__.iteritems() 
                if key not in ['Xsol', 'Xsol_func']}


    def eval_temp(self, Ydata, Xdata):
        Xgen = self.lat_ev_model.get_X()
        Nsamps, Tbins = Ydata.shape[0], Ydata.shape[1]
        print self.QInv.eval()
        AA = self.AA.eval({self.Y : Ydata, Xgen : Xdata})
        AA = AA.reshape([Nsamps*Tbins, self.xDim, self.xDim])
        DetAA = np.linalg.eigvals(AA)
        print DetAA
        print self.AQInvAPlusQ.eval({self.Y : Ydata, Xgen : Xdata})
        print self.BB.eval({self.Y : Ydata, Xgen : Xdata})

    
    def eval_TheChol(self, Ydata, Xdata):
        """
        """
        Xgen = self.lat_ev_model.get_X()
        return ( self.TheChol[0].eval({self.Y : Ydata, Xgen : Xdata}), 
                 self.TheChol[1].eval({self.Y : Ydata, Xgen : Xdata}) )
    
    
    def eval_postX(self, Ydata, Xdata):
        """
        """
        Xgen = self.lat_ev_model.get_X()
        return self.postX.eval({self.Y : Ydata, Xgen : Xdata})
            
    
    def symbsample_X(self, Y=None, X=None):
        """
        TODO: Write docstring
        """
        if Y is None: Y = self.Y
        if X is None: X = self.lat_ev_model.get_X()
        Xgen = self.lat_ev_model.get_X()
        Nsamps, Tbins = Y.shape[0], Y.shape[1]
        
        TheChol = theano.clone(self.TheChol, replace={self.Y : Y, Xgen : X})
        postX = theano.clone(self.postX, replace={self.Y : Y, Xgen : X})
        normSamps = srng.normal([Nsamps, Tbins, self.xDim])
        
        noise, _ = theano.scan(lambda tc1, tc2, ns : 
                               blk_chol_inv(tc1, tc2, ns, lower=False, transpose=True), 
                               sequences=(TheChol[0], TheChol[1], normSamps))
        return postX + noise


    def sample_noise(self, Ydata, Xdata, Nsamps=1):
        Xgen = self.lat_ev_model.get_X()
        N, T = Xdata.shape[0], Xdata.shape[1]
        normSamps = srng.normal([N, T, self.xDim])
        
        TheChol = self.TheChol
        
        noise, _ = theano.scan(lambda tc1, tc2, ns : 
                               blk_chol_inv(tc1, tc2, ns, lower=False, transpose=True), 
                               sequences=(TheChol[0], TheChol[1], normSamps))
        noise_vals = np.zeros((Nsamps, N, T, self.xDim))
        for samp in range(Nsamps):
            noise_vals[samp] = noise.eval({self.Y : Ydata, Xgen : Xdata})
            
        return noise_vals
  

    def sample_X(self, Ydata, Xdata):
        Xgen = self.lat_ev_model.get_X()
        return self.symbsample_X().eval({self.Y : Ydata, Xgen : Xdata})
    
    
    def symbsample_Xalt(self, Y=None, X=None):
        if Y is None: Y = self.Y
        if X is None: X = self.lat_ev_model.get_X()
        Xgen = self.lat_ev_model.get_X()
        Nsamps, Tbins = Y.shape[0], Y.shape[1]
        
        TheChol = theano.clone(self.TheChol, replace={self.Y : Y, Xgen : X})
        postX = theano.clone(self.postX, replace={self.Y : Y, Xgen : X})
        normSamps = srng.normal([Nsamps, Tbins, self.xDim])
        
        noise, _ = theano.scan(lambda tc1, tc2, ns : 
                               blk_chol_inv(tc1, tc2, ns, lower=False, transpose=True), 
                               sequences=(TheChol[0], TheChol[1], normSamps))
        return postX, noise
        


    def compute_Entropy(self, Y=None, X=None):
        if Y is None: Y = self.Y
        if X is None: X = self.X
        Xgen = self.lat_ev_model.get_X()
        Nsamps, Tbins = Y.shape[0], Y.shape[1]
        
        LnDeterminant = theano.clone(self.LnDeterminant, replace={self.Y : Y, Xgen : X})
        Entropy = 0.5*LnDeterminant + 0.5*Nsamps*Tbins*(1 + np.log(2*np.pi))*self.xDim  # Yuanjun has xDim here so I put it but I don't think this is right.
        return Entropy


    def eval_Entropy(self, Ydata, Xdata):
        Xgen = self.lat_ev_model.get_X()
        Nsamps = Ydata.shape[0]
        return self.compute_Entropy().eval({self.Y : Ydata, Xgen : Xdata})/Nsamps
    
    
    def get_Params_Entropy(self):
        return lasagne.layers.get_all_params(self.NNLambdaX) + self.lat_ev_model.get_Params_Evolve() 
    
    
    def get_Params(self):
        return lasagne.layers.get_all_params(self.NNMuX) + lasagne.layers.get_all_params(self.NNLambdaX) if self.common_lat else \
            lasagne.layers.get_all_params(self.NNMuX) + lasagne.layers.get_all_params(self.NNLambdaX) + self.lat_ev_model.get_ParamsEntropy() 

    
    def get_pickledRec(self):
        return {'MuX' : self.MuX, 'Lambda' : self.Lambda}