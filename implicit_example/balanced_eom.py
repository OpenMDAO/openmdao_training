import numpy as np

import openmdao.api as om


class BalancedEOM(om.ImplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input(name='mass',
                       val=np.ones(nn),
                       desc='aircraft mass',
                       units='kg')

        self.add_input(name='velocity',
                       val=np.ones(nn),
                       desc='aircraft velocity magnitude',
                       units='m/s')

        self.add_input(name='lift',
                       val=np.zeros(nn),
                       desc='lift',
                       units='N')

        self.add_input(name='drag',
                       val=np.zeros(nn),
                       desc='drag',
                       units='N')

        self.add_input(name='gamma',
                       val=np.ones(nn)*0.05,
                       desc='flight path angle',
                       units='rad')

        self.add_output(name='alpha',
                       val=np.ones(nn)*0.1,
                       desc='angle of attack',
                       units='rad',
                       lower=-1.0, upper=1.0)
                        
        self.add_output(name='thrust',
                       val=np.ones(nn) * 1.e4,
                       desc='thrust',
                       units='N',
                       lower=10., upper=1.e6)
                       
                       
        self.declare_partials('*', '*', method='fd')

    def apply_nonlinear(self, inputs, outputs, residuals):
        g = 9.80665
        mass = inputs['mass']
        lift = inputs['lift']
        drag = inputs['drag']
        velocity = inputs['velocity']
        gamma = inputs['gamma']
        thrust = outputs['thrust']
        alpha = outputs['alpha']

        residuals['alpha'] = thrust * np.cos(alpha) - drag - mass * g * np.sin(gamma)
        residuals['thrust'] = thrust * np.sin(alpha) + lift - mass * g * np.cos(gamma)
        
    def linearize(self, inputs, outputs, partials):
        g = 9.80665
        mass = inputs['mass']
        lift = inputs['lift']
        drag = inputs['drag']
        velocity = inputs['velocity']
        gamma = inputs['gamma']
        thrust = outputs['thrust']
        alpha = outputs['alpha']
        
        partials['alpha', 'thrust'] = np.cos(alpha)
        partials['alpha', 'alpha'] = thrust * -np.sin(alpha)
        partials['alpha', 'drag'] = -1.
        partials['alpha', 'mass'] = -g * np.sin(gamma)
        partials['alpha', 'gamma'] = -mass * g * np.cos(gamma)
        
        partials['thrust', 'thrust'] = np.sin(alpha)
        partials['thrust', 'alpha'] = thrust * np.cos(alpha)
        partials['thrust', 'lift'] = 1.
        partials['thrust', 'mass'] = -g * np.cos(gamma)
        partials['thrust', 'gamma'] = mass * g * np.sin(gamma)
        
        
    
if __name__ == "__main__":

    prob = om.Problem()
    prob.model = BalancedEOM(num_nodes=1)
    
    prob.model.nonlinear_solver = om.NewtonSolver()
    prob.model.linear_solver = om.DirectSolver()
    
    prob.set_solver_print(level=2)
    
    prob.setup(force_alloc_complex=True)
    
    prob.run_model()
    
    prob.check_partials(method='cs', step=1e-40, compact_print=True)
