class ConnectExample(Group):

	def setup(self):
		""" set some input values - optimizers act on independent variables """
		indeps = self.add_subsystem('indeps', IndepVarComp(), promotes_outputs=['h','U'])
		indeps.add_output('h',10000, units='ft')
		indeps.add_output('U',200, units='kn')
		indeps.add_output('Sref',200, units='ft**2')

		""" add your disciplinary models to the group """
		self.add_subsystem('atmos', StdAtmComp(), promotes_inputs=['h'], promotes_outputs=['rho'])
		self.add_subsystem('lift', ComputeLift(num_pts = 1), promotes_inputs=['rho','U'])
		self.connect('indeps.Sref', 'lift.Sref')






