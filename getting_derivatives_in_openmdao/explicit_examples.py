from __future__ import division
import numpy as np
import openmdao.api as om


class ComputeLift(om.ExplicitComponent):
    
    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('CL', val=0.5, shape=nn, desc='coefficient of lift', units=None)
        self.add_input('rho', val=1.2, shape=nn, desc='air density', units='kg/m**3')
        self.add_input('velocity', val=100., shape=nn, desc='aircraft velocity', units='m/s')
        self.add_input('S_ref', val=8., shape=nn, desc='wing reference area', units='m**2')

        self.add_output('lift', val=np.zeros(nn), desc='aircraft Lift', units='N')
        
        self.declare_partials('*', '*', method='cs')
      
    def compute(self, inputs, outputs):
        CL = inputs['CL']
        rho = inputs['rho']
        velocity = inputs['velocity']
        S_ref = inputs['S_ref']
        
        outputs['lift'] = CL * 0.5 * rho * velocity**2 * S_ref


if __name__ == "__main__":
    
    prob = om.Problem()
    
    prob.model = ComputeLift()
    
    prob.setup()
    prob.run_model()
    
    print('Computed lift: {} Newtons'.format(prob['lift'][0]))
