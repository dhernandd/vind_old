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

import numpy as np

from LatEvModels import LocallyLinearEvolution



class LocallyLinearEvolutionTest(unittest.TestCase):
    """
    """
    optpars = {}
    xDim = 2
    xdata1 = np.array([[0.0, 0.0], [1.0, 1.0], [10.0, 15.0], [50.0, 50.0]])
    Xdata1 = np.array([[[0.0, 0.0], [1.0, 1.0], [50.0, 50.0], [10.0, 15.0]]])
    
    lm = LocallyLinearEvolution(optpars, xDim)
    
    Xdata2 = lm.sample_X(5, 10, withInflow=False)
    
    def test_simple(self):
        print 'QChol:', self.lm.eval_QChol()
        print 'Q:', self.lm.eval_Q()
        print 'alpha:', self.lm.eval_alpha()
        print 'Alinear', self.lm.eval_Alinear()
        
    def test_eval_A(self):
        print 'A wout inflow:', self.lm.eval_A(self.xdata1, withInflow=False)
        print 'A with inflow:', self.lm.eval_A(self.xdata1, withInflow=True)
        
    def test_eval_totalB(self):
        print 'totalB:', self.lm.eval_totalB(self.Xdata1)
        
    def test_eval_totalA(self):
        print 'totalA:', self.lm.eval_totalA(self.Xdata1, withInflow=False)
        print 'totalA:', self.lm.eval_totalA(self.Xdata1, withInflow=True)
        
    def test_eval_nextX(self):
        print 'Next X:', self.lm.eval_nextX(self.Xdata1, withInflow=False)
        print 'Next X (with Inflow):', self.lm.eval_nextX(self.Xdata1, withInflow=True)
        
    def test_runForward_X(self):
        print 'Paths wout Inflow:', self.lm.runForward_X(self.xdata1, 10, withInflow=False)
        
    def test_sampleX(self):
        print 'Paths wout Inflow:', self.lm.sample_X(Nsamps=5, Tbins=10, withInflow=False)
        
    def test_eval_LogDensity_Xterms(self):
        print 'LD, Xterms:', self.lm.eval_LogDensity_Xterms(self.Xdata2)
        
        
if __name__ == '__main__':
    suiteLLE = unittest.TestSuite()
    suiteLLE.addTest(LocallyLinearEvolutionTest("test_simple"))                         # OK!
    suiteLLE.addTest(LocallyLinearEvolutionTest("test_eval_A"))                         # OK
    suiteLLE.addTest(LocallyLinearEvolutionTest("test_eval_totalB"))                    # OK!
    suiteLLE.addTest(LocallyLinearEvolutionTest("test_eval_totalA"))                    # OK!
    suiteLLE.addTest(LocallyLinearEvolutionTest("test_eval_nextX"))                     # OK!
    suiteLLE.addTest(LocallyLinearEvolutionTest("test_runForward_X"))                   # OK!
    suiteLLE.addTest(LocallyLinearEvolutionTest("test_sampleX"))                        # OK!
    suiteLLE.addTest(LocallyLinearEvolutionTest("test_eval_LogDensity_Xterms"))         # OK!

    runner = unittest.TextTestRunner()
    
    runner.run(suiteLLE)
