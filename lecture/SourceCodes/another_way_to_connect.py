class PromoteExample(Group):

	def setup(self):
		""" set some input values - optimizers act on independent variables """
		indeps = self.add_subsystem('indeps', IndepVarComp(), promotes_outputs=['h','U','Sref'])
		indeps.add_output('h', 10000, units='ft')
		indeps.add_output('U', 200, units='kn')
		indeps.add_output('Sref', 200, units='ft**2')

		""" add your disciplinary models to the group """
		self.add_subsystem('atmos', StdAtmComp(), promotes_inputs=['h'], promotes_outputs=['rho'])
		self.add_subsystem('lift', ComouteLift(num_pts = 1), promotes_inputs=['rho','Sref','U'])

if __name__ == '__main__':
	prob = Problem()
	prob.model = PromoteExample()
	prob.setup()
	""" both of these are correct """
	prob['indeps.U'] = 150.
	prob['U'] = 150.
	prob.run_model()
	print(prob['lift.L'])