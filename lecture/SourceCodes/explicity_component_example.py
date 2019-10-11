class ComputeLift(ExplicitComponent):
		"""Compute lift on a wing"""

		def initialize(self):
			self.options.declare('num_pts', default=1, desc="n analysis pts")

		def setup(self):
			npts = self.options['num_pts']
			#Inputs
			self.add_input('Sref', 10.0, units='m**2', desc='Wing ref area')
			self.add_input('rho', 1.225, units='m/s', shape=(npts, ), desc='Air density')
			self.add_input('U', 80.0, units='m/s', shape=(npts, ), desc='Airspeed')
			self.add_input('CL', 0.5, units=None, shape=(npts, ), desc='Lift coefficient')

			# Outputs
			self.add_outputs('L', 0.0, units='N', shape=(npts, ), desc='Lift force')

			# Partial derivatives
			self.declare_partials('L', ['*'])

		def compute(self, inputs, outputs):

			q=1/2 * inputs['rho'] * inputs['U'] ** 2
			outputs['L'] = q * inputs['Sref'] * inputs['CL']