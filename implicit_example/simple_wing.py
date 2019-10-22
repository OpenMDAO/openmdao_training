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
        
if __name__ == '__main__':
    from openmdao.api import Problem, Group, IndepVarComp

    prob = Problem()
    prob.model = Group()
    des_vars = prob.model.add_subsystem('des_vars', IndepVarComp(), promotes=['*'])

    nn = 51

    des_vars.add_output('velocity', 200. * np.ones(nn), units='m/s')
    des_vars.add_output('rho', 1.2 * np.ones(nn), units='kg/m**3')
    des_vars.add_output('S_ref', 594720 * np.ones(nn), units='inch**2')
    des_vars.add_output('alpha', np.linspace(-.2, 0.5, nn))

    prob.model.add_subsystem('SimpleWing', SimpleWing(num_nodes=nn), promotes=['*'])

    prob.setup(check=False, force_alloc_complex=True)
    
    prob.run_model()

    prob.check_partials(compact_print=True, method='fd')
    
    
    prob.model.list_outputs(print_arrays=True)

    import matplotlib.pyplot as plt
    
    plt.plot(prob['drag'], prob['lift'])
    plt.xlabel('drag')
    plt.ylabel('lift')
    plt.show()

    plt.plot(prob['alpha'], prob['lift'])
    plt.xlabel('alpha')
    plt.ylabel('lift')
    plt.show()

    plt.plot(prob['alpha'], prob['drag'])
    plt.xlabel('alpha')
    plt.ylabel('drag')
    plt.show()
