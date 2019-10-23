from __future__ import print_function, division, absolute_import

import numpy as np
from openmdao.api import ExplicitComponent

class SimpleWing(ExplicitComponent):
    
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('alpha', shape=nn, desc='angle of attack', units='rad')
        self.add_input('rho', shape=nn, desc='air density', units='kg/m**3')
        self.add_input('velocity', shape=nn, desc='aircraft velocity', units='m/s')
        self.add_input('S_ref', shape=nn, desc='wing area', units='m**2')

        self.add_output('lift', val=np.zeros(nn), desc='aircraft Lift', units='N')
        self.add_output('drag', val=np.zeros(nn), desc='aircraft drag', units='N')
        
        self.declare_partials('*', '*', method='cs')
      
    def compute(self, inputs, outputs):
        alpha = inputs['alpha']
        rho = inputs['rho']
        velocity = inputs['velocity']
        S_ref = inputs['S_ref']
        
        CL = 2 * np.pi * alpha + 0.10
        CD = alpha ** 2 + 0.020
        
        outputs['lift'] = CL * 0.5 * rho * velocity**2 * S_ref
        outputs['drag'] = CD * 0.5 * rho * velocity**2 * S_ref
        
