class PromoteExample(Group):

	def setup(self):
		""" set some input values - optimizers act on independent variables """
		indeps = self.add_subsystem('indeps', IndepVarComp(), promotes_outputs=['h','U','Sref'])
		indeps.add_output('h', 10000, units='ft')
		indeps.add_output('U', 200, units='kn')
		indeps.add_output('Sref', 200, units='ft**2')
		indeps2= self.add_subsystem('indeps2', IndepVarComp(), promotes_outputs=['U'])
		indeps2.add_output('U', 200, units='kn')

		""" add your disciplinary models to the group """
		self.add_subsystem('atmos', StdAtmComp(), promotes_inputs=['h'], promotes_outputs=['rho'])
		self.add_subsystem('lift', ComouteLift(num_pts = 1), promotes_inputs=['rho','Sref','U'])
		self.add_subsystem('lift2', ComouteLift(num_pts = 1), promotes_inputs=['rho','Sref','U'])
