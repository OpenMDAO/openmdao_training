from __future__ import print_function, division

from time import time
import numpy as np
from collections import OrderedDict
import pickle

import openmdao.api as om


data = OrderedDict()
data['Analytic Dense'] = None
data['Analytic Sparse'] = None
data['Approximated'] = None
data['Approximated Colored'] = None

with open('timing_data.pkl', 'rb') as f:
    output_data = pickle.load(f)

timing_data = output_data['timing_data']
nns = output_data['nns']

import matplotlib.pyplot as plt

plt.figure()

for i_method, key in enumerate(data):
    plt.loglog(nns, timing_data[:, i_method], label=key)
    
plt.legend()
plt.xlabel('Num nodes')
plt.ylabel('Time to compute total derivs, secs')
plt.savefig('timing_data.pdf')
