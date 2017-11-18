# -*- coding: utf-8 -*-
# Copyright 2017 Daniel Hernandez Diaz, Columbia University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================
import numpy as np

import theano
import theano.tensor as T
import theano.tensor.nlinalg as Tnla
import lasagne
from lasagne.nonlinearities import softmax, linear, softplus
from lasagne.layers import InputLayer, DenseLayer
from lasagne.init import Orthogonal, Normal, Uniform


def flow_modulator(x, x0=30.0, a=0.08):
    return (1 - np.tanh(a*(x - x0)))/2


class LinearEvolution(object):
    """
    TODO: Implement Linear Evolution here a la Archer-Gao. 
    
    Note that this is strictly unnecessary since linear evolution can be
    simulated setting alpha=0
    """
    def __init__(self):
        """
        """
        pass
    
    
class LocallyLinearEvolution(object):
    """
    The Locally Linear Evolution Model:
        x_{t+1} = A(x_t)x_t + eps
    where eps is Gaussian noise.
    
    An evolution model, it should implement the following key methods:
        sample_X:    Draws samples from the Markov Chain.
        compute_LogDensity_Xterms: Computes the loglikelihood of the chain.
    """
    def __init__(self, LatPars, xDim, X=None, nnname='Ev'):
        """
        Args:
            LatPars:
            xDim:
            X:
            nnname:
        """
        for key in LatPars.keys():
            setattr(self, key, LatPars[key])
        
        self.x = T.matrix('x')
        self.X = T.tensor3('X') if X is None else X
        self.Nsamps = Nsamps = self.X.shape[0]
        self.nsamps = nsamps = self.x.shape[0]
        self.Tbins = Tbins = self.X.shape[1]
        self.xDim = xDim
        
        if not hasattr(self, 'QInvChol'):
            self.QInvChol = theano.shared(4.0*np.eye(xDim), 'QInvChol')
        self.QChol = Tnla.matrix_inverse(self.QInvChol)
        self.QInv = T.dot(self.QInvChol, self.QInvChol.T)
        self.Q = Tnla.matrix_inverse(self.QInv)
        
        if not hasattr(self, 'Q0InvChol'):
            self.Q0InvChol = theano.shared(2.0*np.eye(xDim), 'Q0InvChol')
        self.Q0Chol = Tnla.matrix_inverse(self.Q0InvChol)
        self.Q0Inv = T.dot(self.Q0InvChol, self.Q0InvChol.T)
        
        self.x0 = theano.shared(np.zeros(xDim), 'x0')
        
        # Define base linear element of the evolution 
        if not hasattr(self, 'Alinear'):
            self.Alinear = theano.shared(value=np.eye(xDim), name='Alinear', borrow=True)

        evnodes = 60
        self.optimize_alpha = False
        if not hasattr(self, 'alpha'):
            self.alpha = alpha = (0.1 if not self.optimize_alpha else 
                                  theano.shared(value=0.2, name='alpha', borrow=True) )
        else:
            alpha = self.alpha
        if not hasattr(self, 'NNEvolve'):
            NNEvolve = InputLayer((None, self.xDim), name=nnname+'_IL')
            NNEvolve = DenseLayer(NNEvolve, evnodes, nonlinearity=softmax, W=Orthogonal(), 
                                  name=nnname+'_HL1')
            self.NNEvolve = DenseLayer(NNEvolve, xDim**2, nonlinearity=linear, W=Uniform(0.9), 
                                       name=nnname+'_OL')
        NNEvolve_out = lasagne.layers.get_output(self.NNEvolve, inputs=self.x)
        self.B = NNEvolve_out.reshape([nsamps, xDim, xDim])
        
        # Only t up to T-1 are necessary to find the evolution matrix. 
        B_input = self.X[:,:-1,:].reshape([self.Nsamps*(self.Tbins-1), self.xDim])
        Bs = lasagne.layers.get_output(self.NNEvolve, inputs=B_input)        
        Bs = Bs.reshape([Nsamps*(Tbins-1), xDim, xDim])
        
        # Compute the gradients of B.
        # TODO: Comment all this so that you know what you are doing.
        # TODO: Actually implement this thing. Major modifications in the code required.
        def _compute_B_grads(self):
            flatBs_NTm1dd = Bs.flatten()
            
            # The problem is that this function computes the gradient w.r.t. all
            # the elements in self.X. This is absolutely unnecessary - the
            # entries in B only depend on one particular X_t - and
            # unfortunately, it turns a computation that should be O(N) in one
            # that is O(N^2*T), obviously increasing the running time like
            # hell. So it seems I will have to do all this gradient stuff old
            # school.
            grad_flatBs_NTm1ddxNxTxd, _ = theano.scan(lambda i, b, x: T.grad(b[i], x), 
                                                sequences=T.arange(flatBs_NTm1dd.shape[0]),
                                                non_sequences=[flatBs_NTm1dd, self.X])
            grad_flatBs_NTm1ddxNxTm1xd = grad_flatBs_NTm1ddxNxTxd[:,:,:-1,:]
            grad_flatBs_NTm1xddxNTm1xd = grad_flatBs_NTm1ddxNxTm1xd.reshape([Nsamps*(Tbins-1),
                                                                xDim**2, Nsamps*(Tbins-1), xDim])
            grad_flatBs_NTm1xdxdxd, _ = theano.scan(lambda i, X: X[i,:,i,:].reshape([xDim, xDim, xDim]).dimshuffle(2, 0, 1), 
                            sequences=T.arange(Nsamps*(Tbins-1)),
                            non_sequences=[grad_flatBs_NTm1xddxNTm1xd])
            self.totalgradBs = grad_flatBs_NTm1xdxdxd.reshape([Nsamps, Tbins-1, xDim, xDim, xDim])
    #         This works.
    #         sampleX = np.random.rand(80, 30, 2)
    #         print 'Finished gradients'
    #         print 'GradBB:', self.totalgradBs.eval({self.X : sampleX})
    #         T1 = self.alpha*T.dot(self.totalgradBs.dimshuffle(0,1,2,4,3), T.dot(self.QInv, self.Alinear))
    #         T1_func = theano.function([self.X], T1)
    #         print 'T1:', T1_func(sampleX)
    #         sampleX2 = np.random.rand(80, 30, 2)
    #         print 'T1a:', T1_func(sampleX2)
        
        self.A = self.Alinear + alpha*self.B
        As, _ = theano.scan(fn=lambda Bt : self.Alinear + alpha*Bt, sequences=Bs)

        self.totalB = T.reshape(Bs, [Nsamps, Tbins-1, xDim, xDim])
        self.totalA = T.reshape(As, [Nsamps, Tbins-1, xDim, xDim])

        
    def get_X(self):
        return self.X
    

    def get_QInv(self):
        """
        Useful with external recognition models
        """
        return self.QInv
    

    def get_Q0Inv(self):
        """
        Useful with external recognition models
        """
        return self.Q0Inv
    
    
    def eval_QChol(self):
        return self.QChol.eval()
    
    def eval_Q(self):
        return self.Q.eval()

    def eval_xDim(self):
        return self.xDim

    def eval_Alinear(self):
        return self.Alinear.eval()
                

    def get_A(self):
        return self.A

    
    def get_totalA(self):
        return self.totalA
        
    
    def eval_alpha(self):
        return self.alpha.eval() if self.optimize_alpha else self.alpha
    
    
    def eval_A(self, xdata, withInflow=False):
        Nsamps = xdata.shape[0]
        xdatanorms = np.linalg.norm(xdata, axis=1)
        
        A = self.A.eval({self.x : xdata})
        A_rshpd = A.reshape(Nsamps, self.xDim**2)
        AwInflow = (flow_modulator(xdatanorms)*A_rshpd.T).T.reshape(Nsamps, self.xDim, self.xDim) + \
                     0.9*np.tensordot((1.0 - flow_modulator(xdatanorms)), np.eye(self.xDim), 0)  
                     # The Reshape Master they call me.
        
        return A if not withInflow else AwInflow
    
    
    def eval_totalB(self, Xdata):
        """
        Given Xdata of the hidden space, computes the matrices B(Xdata) that
        implement the evolution in the hidden space
            x_{t+1} = B(x_t)x_t
         
        Inputs:
            Xdata -> Latent space sample (numeric)
        """
        return self.totalB.eval({self.X : Xdata})
    
     
    def eval_totalA(self, Xdata, withInflow=False):
        """
        Given Xdata of the hidden space, computes the matrices A(Xdata) that
        implement the evolution in the hidden space
            x_{t+1} = A(x_t)x_t
         
        Inputs:
            Xdata: Latent space sample (numeric) 
            withInflow: When set to True, this bool variable imposes an inward
            flow coming from infinity on A(Z) that dominates for large |Z|. It
            should be set to True only for generating fake data, where we want
            to ensure that latent paths do not blow up. When fitting, always set
            to False.
        """
        Nsamps, Tbins = Xdata.shape[0], Xdata.shape[1]
        Xdatanorms = np.linalg.norm(Xdata[:,:-1,:], axis=2).reshape(Nsamps*(Tbins-1))
        
        totalA = self.totalA.eval({self.X : Xdata})
        if not withInflow:
            return totalA
        else:
            totalA_rshpd = totalA.reshape(Nsamps*(Tbins-1), self.xDim**2)
            
            A1 = flow_modulator(Xdatanorms)*totalA_rshpd.T
            A2 = np.tensordot((1.0 - flow_modulator(Xdatanorms).reshape(Nsamps, Tbins-1)), 
                              np.eye(self.xDim), 0)
            totalAwInflow = A1.T.reshape(Nsamps, Tbins-1, self.xDim, self.xDim) + 0.9*A2  
            # The Shapeshifter, they call me.
        
            return totalAwInflow
        
     
    def eval_nextX(self, Xdata, withInflow=False):
        """
        Given a symbolic array of points in latent space Xdata = [X0, X1,...,XT], \
        gives the prediction for the next time point
         
        This is useful for plotting.
        
        Args:
            Xdata: Points in latent space at which the dynamics A(X) shall be
            determined.
            withInflow: Should an inward flow from infinity be superimposed to A(X)?
        """
        Nsamps, Tbins = Xdata.shape[0], Xdata.shape[1]
        
        A = self.eval_totalA(Xdata, withInflow).reshape([Nsamps*(Tbins-1), self.xDim, self.xDim])
        Xdata = Xdata[:,:-1,:].reshape(Nsamps*(Tbins-1), self.xDim)
                
        return np.einsum('ij,ijk->ik', Xdata, A).reshape(Nsamps, Tbins-1, self.xDim)
     
     
    def runForward_X(self, X0, Tbins=100, withInflow=False):
        """
        Runs forward the deterministic hidden dynamics.
        
        
        """
        Nsamps = X0.shape[0]
        X_vals = np.zeros([Nsamps, Tbins, self.xDim])
        X_vals[:,0,:] = X0
        
        for curr_tbin in range(Tbins-1):
            A = self.eval_A(X_vals[:,curr_tbin,:], withInflow)
            X_vals[:,curr_tbin+1,:] = np.einsum('ij,ijk->ik', X_vals[:,curr_tbin,:], A)

        return X_vals
         
     
    def sample_X(self, Nsamps=30, Tbins=100, X0data=None, 
                 withInflow=False, path_mse_threshold=1.0):
        """
        Runs forward the stochastic model for the latent space.
         
        Returns a numpy array of samples
        """
        Q0Chol = self.Q0Chol.eval()
        QChol = self.QChol.eval()
        Nsamps = X0data.shape[0] if X0data is not None else Nsamps
        Xdata = np.zeros([Nsamps, Tbins, self.xDim])
        x0scale = 25.0

        print('alpha:', self.alpha)
        for samp in range(Nsamps):
            samp_norm = 0.0 # use to avoid paths that start too close to an attractor
            while samp_norm < path_mse_threshold: # lower path_mse_threshold to keep paths closer to trivial trajectories, x = const.
                Xsamp = np.zeros([Tbins, self.xDim])
                x0 = ( x0scale*np.dot(np.random.randn(self.xDim), Q0Chol) if X0data is None 
                      else X0data[samp] )
                Xsamp[0] = x0
                
                noise_samps = np.random.randn(Tbins, self.xDim)                                        
                for curr_tbin in range(Tbins-1):
                    A = np.reshape(self.eval_A(np.reshape(Xsamp[curr_tbin], 
                                [1, self.xDim]), withInflow), [self.xDim, self.xDim])
#                     print(self.eval_A(np.reshape(Xsamp[curr_tbin], [1, self.xDim]), withInflow))
                    # X_{t+1} = A(X_t)·X_t + eps 
                    Xsamp[curr_tbin+1] = ( np.dot(Xsamp[curr_tbin], A) + 
                                           np.dot(noise_samps[curr_tbin+1], QChol) )
#                     import time
#                     time.sleep(2)
                
                # Compute MSE and discard path is MSE < path_mse_threshold (trivial paths)
                Xsamp_mse = np.mean([np.linalg.norm(Xsamp[tbin+1] - Xsamp[tbin]) 
                                     for tbin in range(Tbins-1)])
                samp_norm = Xsamp_mse
            
            Xdata[samp,:,:] = Xsamp

        return Xdata
 
 
    def get_Params(self):
        return ( [self.QInvChol] + [self.Q0InvChol] + [self.x0] + 
                 lasagne.layers.get_all_params(self.NNEvolve) + [self.Alinear] + [self.alpha] 
                 if self.optimize_alpha else [self.QInvChol] + [self.Q0InvChol] + [self.x0] +
                 lasagne.layers.get_all_params(self.NNEvolve) + [self.Alinear] ) 
    

    def get_ParamsEntropy(self):
        return ( [self.QInvChol] + lasagne.layers.get_all_params(self.NNEvolve) + 
            [self.Alinear] + [self.alpha] if self.optimize_alpha else [self.QInvChol] + 
             lasagne.layers.get_all_params(self.NNEvolve) + [self.Alinear] )

    
    def compute_LogDensity_Xterms(self, X=None, Xprime=None, padleft=False, persamp=False):
        """
        Computes the symbolic log p(X, Y).
        p(X, Y) is computed using Bayes Rule. p(X, Y) = P(Y|X)p(X).
        p(X) is normal as described in help(PNLDS).
        p(Y|X) is py with output self.output(X).
         
        Inputs:
            X : Symbolic array of latent variables.
            Y : Symbolic array of X
         
        NOTE: This function is required to accept symbolic inputs not necessarily belonging to the class.
        """
        if X is None:
            X = self.X
        if padleft:
            X = T.shape_padleft(X, 1)

        Nsamps, Tbins = X.shape[0], X.shape[1]

        totalApred = theano.clone(self.totalA, replace={self.X : X})
        totalApred = T.reshape(totalApred, [Nsamps*(Tbins-1), self.xDim, self.xDim])
        Xprime = T.batched_dot(X[:,:-1,:].reshape([Nsamps*(Tbins-1), self.xDim]), totalApred) if Xprime is None else Xprime
        Xprime = T.reshape(Xprime, [Nsamps, Tbins-1, self.xDim])

        resX = X[:,1:,:] - Xprime
        resX0 = X[:,0,:] - self.x0
        
        # L = -0.5*(∆X_0^T·Q0^{-1}·∆X_0) - 0.5*Tr[∆X^T·Q^{-1}·∆X] + 0.5*N*log(Det[Q0^{-1}])
        #     + 0.5*N*T*log(Det[Q^{-1}]) - 0.5*N*T*d_X*log(2*Pi)
        L1 = -0.5*(resX0*T.dot(resX0, self.Q0Inv)).sum()
        L2 = -0.5*(resX*T.dot(resX, self.QInv)).sum()
        L3 = 0.5*T.log(Tnla.det(self.Q0Inv))*Nsamps
        L4 = 0.5*T.log(Tnla.det(self.QInv))*(Tbins-1)*Nsamps
        L5 = -0.5*(self.xDim)*np.log(2*np.pi)*Nsamps*Tbins
        LatentDensity = L1 + L2 + L3 + L4 + L5
                
        return LatentDensity, L1, L2, L3, L4, L5


    def eval_LogDensity_Xterms(self, Xdata, padleft=False, persamp=False):
        LD = self.compute_LogDensity_Xterms(padleft=padleft, persamp=persamp)
        Nsamps = Xdata.shape[0]
        return LD[0].eval({self.X : Xdata})/Nsamps, LD[1].eval({self.X : Xdata})/Nsamps, LD[2].eval({self.X : Xdata})/Nsamps, \
            LD[3].eval({self.X : Xdata})/Nsamps, LD[4].eval({self.X : Xdata})/Nsamps, LD[5].eval({self.X : Xdata})/Nsamps


    def defineLattice(self, x1range=(-30.0, 30.0), x2range=(-30.0, 30.0)):
        x1coords = np.linspace(x1range[0], x1range[1])
        x2coords = np.linspace(x2range[0], x2range[1])
        Xlattice = np.array(np.meshgrid(x1coords, x2coords))
        return Xlattice.reshape(self.xDim, -1).T


    def quiver2D_flow(self, data=None, clr='black', scale=50, nlds=True, 
                      x1range=(-35.0, 35.0), x2range=(-35.0, 35.0), figsize=(13,13), 
                      pause=True, draw=True, withInflow=False, newfig=True, savefile=None):
        """
        TODO: Write the docstring for this bad boy.
        """
        import matplotlib.pyplot as plt
        if newfig:
            plt.ion()
            plt.figure(figsize=figsize)
        if data is None:
            lattice = self.defineLattice(x1range, x2range)
            Tbins = lattice.shape[0]
            lattice = np.reshape(lattice, [1, Tbins, self.xDim])
            nextX = self.eval_nextX(lattice, withInflow=withInflow).reshape(Tbins-1, self.xDim)
            X = lattice[:,:-1,:].reshape(Tbins-1, self.xDim)

            plt.quiver(X.T[0], X.T[1], nextX.T[0]-X.T[0], nextX.T[1]-X.T[1], color=clr, scale=scale)
            axes = plt.gca()
            axes.set_xlim(x1range)
            axes.set_ylim(x2range)
            if draw: plt.draw()  
            if pause:
                plt.pause(0.001)
                raw_input('Press Enter to continue.')
            if savefile is not None:
                plt.savefig(savefile)
            else:
                pass

            
    def plot2D_sampleX(self, X0data=np.random.rand(1, 2), Xdata=None, figsize=(13,13), newfig=True, pause=True, draw=True, skipped=1):
        """
        Plots the evolution of the dynamical system in a 2D projection.
         
        """
        import matplotlib.pyplot as plt
        
        ctr = 0
        if newfig:
            plt.ion()
            plt.figure(figsize=figsize)
        if Xdata is None:
            Xdata = self.sample_X(X0data)
#             print Xdata
            self.quiver2D_flow(draw=False, pause=False)
            for samp in Xdata:
                if ctr % skipped == 0:
                    plt.plot(samp[:,0], samp[:,1])
                    plt.plot(samp[0,0], samp[0,1], 'o')
                    axes = plt.gca()
                ctr += 1
            if draw: plt.draw()  
            if pause:
                plt.pause(0.001)
                raw_input('Press Enter to continue.')
            else:
                pass                    
        else:
            for samp in Xdata:
                if ctr % skipped == 0:
                    plt.plot(samp[:,0], samp[:,1], linewidth=2)
                    plt.plot(samp[0,0], samp[0,1], 'bo')
                    axes = plt.gca()
                ctr += 1
            if draw: plt.draw()  
            if pause:
                plt.pause(0.001)
                raw_input('Press Enter to continue.')

            
        return axes


    def plot_2Dquiver_paths(self, Xdata, rslt_file, withInflow=False):
        """
        """
        import matplotlib.pyplot as plt
        axes = self.plot2D_sampleX(Xdata=Xdata, pause=False, draw=False, newfig=True)
        x1range, x2range = axes.get_xlim(), axes.get_ylim()
        s = int(5*max(abs(x1range[0]) + abs(x1range[1]), abs(x2range[0]) + abs(x2range[1]))/3)
        self.quiver2D_flow(pause=False, x1range=x1range, x2range=x2range, scale=s, newfig=False, withInflow=withInflow)
        plt.savefig(rslt_file)
        plt.close()

