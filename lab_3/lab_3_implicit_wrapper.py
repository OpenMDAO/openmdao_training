import numpy as np

import openmdao.api as om

def fmt_data(data): 
    """helper to format array data with lots of sig figs"""     
    to_str = ['{:10.16f}'.format(n) for n in data]
    return '[{}]'.format(','.join(to_str))

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


        self.declare_partials('u', '*', method='fd', step=1e-4, step_calc='abs')


        self.declare_partials('compliance', ['h', 'u'], method='fd', step=1e-4, step_calc='abs')
        # if we look in the wrapper, we can see that this deriv is analytically 1 
        self.declare_partials('compliance', 'compliance', val=1) 

        # note: we know there is no derivative with respect to displacements, so we don't declare it
        self.declare_partials('volume', 'h', method='fd', step=1e-4, step_calc='abs')
        # if we look in the wrapper, we can see that this deriv is analytically 1 
        self.declare_partials('volume', 'volume', val=1)


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
                'h = np.array({})'.format(fmt_data(h))
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
                'h = np.array({})'.format(fmt_data(h)),
                'u = np.array({})'.format(fmt_data(u)),
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

        residuals['u'] = data['u_residuals']
        residuals['compliance'] = data['c_residual']
        residuals['volume'] = data['v_residual']

    

if __name__ == "__main__": 

    NUM_ELEMENTS = 5

    p = om.Problem()

    dvs = p.model.add_subsystem('dvs', om.IndepVarComp(), promotes=['*'])
    dvs.add_output('h', val=np.ones(NUM_ELEMENTS)*1.0) 
    p.model.add_subsystem('FEM', FEMBeam(E=1, L=1, b=0.1, 
                                         num_elements=NUM_ELEMENTS),
                          promotes_inputs=['h'], 
                          promotes_outputs=['compliance', 'volume'])


    p.driver = om.ScipyOptimizeDriver()
    p.driver.options['tol'] = 1e-4
    p.driver.options['disp'] = True
    p.model.add_design_var('h', lower=0.01, upper=10.0)
    p.model.add_objective('compliance')
    p.model.add_constraint('volume', equals=0.01)

    # p.model.approx_totals(method='fd', step=1e-4, step_calc='abs')

    p.model.linear_solver = om.DirectSolver()

    p.setup()

    # p['h'] = [0.14007896, 0.12362061, 0.1046475 , 0.08152954, 0.05012339]

    p.run_driver()

    p.model.list_outputs(print_arrays=True)


