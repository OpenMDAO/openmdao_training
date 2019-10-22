class ConnectExample(Group):

	def setup(self):
		""" set some input values - optimizers act on independent variables """
		indeps = self.add_subsystem('indeps', IndepVarComp())
		indeps.add_output('h',10000, units='ft')
		indeps.add_output('U',200, units='kn')
		indeps.add_output('Sref',200, units='ft**2')

		""" add your disciplinary models to the group """
		self.add_subsystem('atmos', StdAtmComp())
		self.add_subsystem('lift', ComputeLift(num_pts = 1))

		""" connect variables together """
		self.connect('indeps.h','atmos.h')
		self.connect('atmos.rho','lift.rho')
		self.connect('indeps.Sref','lift.Sref')
		self.connect('indeps.U','lift.U')

if__name__=="__main__":
	prop = Problem()
	prob.model = ConnectExample()
	prob.setup()
	prob['indeps.U'] = 150.
	prob.run_model()
	print(prob['lift.L'])






