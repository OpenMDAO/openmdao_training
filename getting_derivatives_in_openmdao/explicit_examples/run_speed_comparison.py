from __future__ import print_function, division

from time import time
import numpy as np
from collections import OrderedDict
import pickle

import openmdao.api as om


nns = [2**i for i in range(12)]
num_nns = len(nns)
num_repeats = 10

data = OrderedDict()
data['Analytic Dense'] = np.zeros((num_nns, num_repeats))
data['Analytic Sparse'] = np.zeros((num_nns, num_repeats))
data['Approximated'] = np.zeros((num_nns, num_repeats))
data['Approximated Colored'] = np.zeros((num_nns, num_repeats))

timing_data = np.zeros((num_nns, len(data)))

for i_method, key in enumerate(data):
    
    print()
    print(i_method, key)    
    
    if key == 'Analytic Dense':
        from compute_lift_analytic_dense import ComputeLift
    if key == 'Analytic Sparse':
        from compute_lift_analytic_sparse import ComputeLift
    if key == 'Approximated':
        from compute_lift_approximated import ComputeLift
    if key == 'Approximated Colored':
        from compute_lift_approximated_colored import ComputeLift
        
    for i_nn, nn in enumerate(nns):
        
        print('Running {} cases of {} num_nodes'.format(num_repeats, nn))
        
        prob = om.Problem(model=om.Group())

        ivc = prob.model.add_subsystem('indep_var_comp', om.IndepVarComp(), promotes=['*'])
        ivc.add_output('CL', val=0.5, shape=nn, units=None)
        ivc.add_output('rho', val=1.2, shape=nn, units='kg/m**3')
        ivc.add_output('velocity', val=100., shape=nn, units='m/s')
        ivc.add_output('S_ref', val=8., shape=nn, units='m**2')

        prob.model.add_subsystem('compute_lift', ComputeLift(num_nodes=nn), promotes=['*'])

        prob.setup()
        prob.run_model()
        
        for i_repeat in range(num_repeats):

            pre_time = time()
            
            prob.compute_totals(['lift'], ['CL', 'rho', 'velocity', 'S_ref'])
            
            post_time = time()
            duration = post_time - pre_time
            
            data[key][i_nn, i_repeat] = duration
            
        timing_data[i_nn, i_method] = np.mean(data[key][i_nn, :])
        
        
output_data = {
    'timing_data' : timing_data,
    'nns' : nns,
    }
    
with open('timing_data.pkl', 'wb') as f:
    pickle.dump(output_data, f)
