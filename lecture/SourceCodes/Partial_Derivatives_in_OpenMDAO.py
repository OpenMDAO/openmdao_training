class WeightBuild(ExplicitComponent)
	"""Compute TOW from component weights"""

def setup(self):
	""" define the following inputs: W_payload, W_empty, TOW"""
	self.add_input('W_payload', 800, units='1lbm')
	self.add_input('W_empty', 5800, units='lbm')
	self.add_input('W_battery', 1500, units='lbm')

	""" define the following outputs: W_battery """
	self.add_output('TOW', val=6000, units='lbm')

	""" declare generic finite difference partials """
	self.declare_partials('TOW',['*'])

def compute(self, inputs, outputs):
	""" implement the calculation W_battery = TOW - W_payload - W_empty """
	outputs['TOW'] = inputs['W_battery'] + inputs['W_payload'] + inputs['W_empty']

def compute_partials (self, inputs, partials):
	partials['TOW', 'W_battery'] = 1
	partials['TOW', 'W_payload'] = 1
	partials['TOW', 'W_empty'] 	 = 1

	