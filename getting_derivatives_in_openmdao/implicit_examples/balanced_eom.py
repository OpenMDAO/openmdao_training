import numpy as np

import openmdao.api as om


# Acceleration due to gravity in m/s**2
g = 9.80665

class BalancedEOM(om.ImplicitComponent):
    """
    An implicit component to solve for the angle of attack and thrust values
    needed to balance the forces that an aircraft experiences in steady flight.
    """

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
                       
        arange = np.arange(nn)
        self.declare_partials('*', '*', rows=arange, cols=arange)

    # Compute the residual values of alpha and thrust
    def apply_nonlinear(self, inputs, outputs, residuals):
        mass = inputs['mass']
        lift = inputs['lift']
        drag = inputs['drag']
        velocity = inputs['velocity']
        gamma = inputs['gamma']
        thrust = outputs['thrust']
        alpha = outputs['alpha']

        residuals['alpha'] = thrust * np.cos(alpha) - drag - mass * g * np.sin(gamma)
        residuals['thrust'] = thrust * np.sin(alpha) + lift - mass * g * np.cos(gamma)
        
    # Compute the partial derivatives of the residual equations wrt each of
    # the inputs
    def linearize(self, inputs, outputs, partials):
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
        
        
