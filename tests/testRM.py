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
import unittest

import sys
sys.path.append('../../')

import numpy as np

import vind

__package__ = 'vind.tests'
from ..code.LatEvModels import LocallyLinearEvolution
from ..code.ObservationModels import PoissonObsTSGM
from ..code.RecognitionModels import SmoothingNLDSTimeSeries



class SmoothingDSTest(unittest.TestCase):
    """
    """
    xDim = 2
    optpars = {}
    xdata1 = np.array([[0.0, 0.0], [1.0, 1.0], [10.0, 15.0], [50.0, 50.0]])
    Xdata1 = np.array([[[0.0, 0.0], [1.0, 1.0], [50.0, 50.0], [10.0, 15.0]]])

    lm = LocallyLinearEvolution(optpars, xDim)
    Xdata2 = lm.sample_X(5, 10, withInflow=False)

    yDim = 10
    obspars = {}
    obsm = PoissonObsTSGM(obspars, yDim, xDim, lat_ev_model=lm)
    Ydata3, Xdata3 = obsm.sample_XY(Nsamps=5, Tbins=10, withInflow=False)
    
    recpars = {}
    rm = SmoothingNLDSTimeSeries(recpars, yDim, xDim, lat_ev_model=lm)

    def test_simple(self):
        print 'MuX:', self.rm.eval_Mu(self.Ydata3[:2]) # OK!
        print 'LambdaCholX', self.rm.eval_LambdaChol(self.Ydata3[:2]) # OK!
        print 'Lambda', self.rm.eval_Lambda(self.Ydata3[:2])

    def test_eval_TheChol(self):
        TC0, TC1 = self.rm.eval_TheChol(self.Ydata3[:2], self.Xdata3[:2])[0]
        print 'Temp0:', TC0, TC1, TC0.shape, TC1.shape 
        
    def test_eval_postX(self):
        print 'PostX:', self.rm.eval_postX(self.Ydata3, self.Xdata3)

    def test_sample_X(self):
        print 'X sample 1:', self.rm.sample_X(self.Ydata3, self.Xdata3)

    def test_compute_Entropy(self):
        print 'Entropy:', self.rm.eval_Entropy(self.Ydata3, self.Xdata3)
        
    def test_sample_noise(self):
        print 'This may take a while....'
        noise_vals = self.rm.sample_noise(self.Ydata3, self.Xdata3)
        print 'Noise sample:', noise_vals[0] 



if __name__ == '__main__':
    suiteSmDS = unittest.TestSuite()
    suiteSmDS.addTest(SmoothingDSTest("test_simple"))                           # OK!
    suiteSmDS.addTest(SmoothingDSTest("test_eval_TheChol"))                     # OK!
    suiteSmDS.addTest(SmoothingDSTest("test_eval_postX"))                       # OK!
    suiteSmDS.addTest(SmoothingDSTest("test_sample_X"))                         # OK!
    suiteSmDS.addTest(SmoothingDSTest("test_compute_Entropy"))                  # OK!
    suiteSmDS.addTest(SmoothingDSTest("test_sample_noise"))

    runner = unittest.TextTestRunner()
    
    runner.run(suiteSmDS)