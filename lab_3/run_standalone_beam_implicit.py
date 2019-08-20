import numpy as np

import openmdao.api as om

class FEMBeam(om.ExternalCodeImplicitComp): 

    def initialize(self):
        self.options.declare('E')
        self.options.declare('L')
        self.options.declare('b')
        self.options.declare('num_elements', int)

    def setup(self): 
        E = self.options['E']
        L = self.options['L']
        b = self.options['b']
        num_elements = self.options['num_elements']
        num_nodes = num_elements + 1

        force_vector = np.zeros(2 * num_nodes)
        force_vector[-2] = -1.

        self.add_input('h', shape=num_elements)
        self.add_output('u', shape=2*num_elements+4)
        self.add_output('compliance', shape=1)        
        self.add_output('volume', shape=1)

        # providing these is optional; the component will verify that any input
        # files exist before execution and that the output files exist after.
        self.options['external_input_files'] = ['input.txt']
        self.options['external_output_files'] = ['output.txt']

        self.options['command_solve'] = ['python', 'standalone_beam.py', 'solve']

        self.options['command_apply'] = ['python', 'standalone_beam.py', 'apply']


    def solve_nonlinear(self, inputs, outputs):
        E = self.options['E']
        L = self.options['L']
        b = self.options['b']
        num_elements = self.options['num_elements']

        h = inputs['h']

        with open('input.txt', 'w') as f: 
            data = [
                'num_elements = {}'.format(num_elements), 
                'E = {}'.format(E), 
                'L = {}'.format(L),
                'b = {}'.format(b), 
                'h = np.array({})'.format(h.tolist())
            ]

            f.write("\n".join(data))

        # method from base class to execute the code with the given command_apply
        super(FEMBeam, self).solve_nonlinear(inputs, outputs)

        with open('output.txt', 'r') as f: 
            data = {}
            # parses the output and puts the variables into the data dictionary
            exec(f.read(), {}, data) 

        outputs['u'] = data['u']
        outputs['compliance'] = data['compliance']
        outputs['volume'] = data['volume']


    def apply_nonlinear(self, inputs, outputs, residuals): 
        E = self.options['E']
        L = self.options['L']
        b = self.options['b']
        num_elements = self.options['num_elements']

        h = inputs['h']

        u = outputs['u']
        compliance = outputs['compliance'][0]
        volume = outputs['volume'][0]

        with open('input.txt', 'w') as f: 
            data = [
                'num_elements = {}'.format(num_elements), 
                'E = {}'.format(E), 
                'L = {}'.format(L),
                'b = {}'.format(b), 
                'h = np.array({})'.format(h.tolist()),
                'u = np.array({})'.format(u.tolist()),
                'compliance = {}'.format(compliance), 
                'volume = {}'.format(volume),
            ]

            f.write("\n".join(data))

        # method from base class to execute the code with the given command_apply

        super(FEMBeam, self).apply_nonlinear(inputs, outputs, residuals)

        with open('output.txt', 'r') as f: 
            data = {}
            # parses the output and puts the variables into the data dictionary
            exec(f.read(), {}, data) 

        print('data', data.keys())

        residuals['u'] = data['u_residuals']
        residuals['compliance'] = data['c_residual']
        residuals['volume'] = data['v_residual']

    

if __name__ == "__main__": 

    p = om.Problem()

    p.model = FEMBeam(E=1, L=1, b=0.1, num_elements=50)

    p.setup()

    p.run_model()
    p.model.run_apply_nonlinear()

    p.model.list_outputs(residuals=True)


