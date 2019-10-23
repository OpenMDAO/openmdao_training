from openmdao.api import Problem, Group, IndepVarComp
import numpy as np

from simple_wing import SimpleWing


prob = Problem()
prob.model = Group()
des_vars = prob.model.add_subsystem('des_vars', IndepVarComp(), promotes=['*'])

nn = 51

des_vars.add_output('velocity', 200. * np.ones(nn), units='m/s')
des_vars.add_output('rho', 1.2 * np.ones(nn), units='kg/m**3')
des_vars.add_output('S_ref', 383.7 * np.ones(nn), units='m**2')
des_vars.add_output('alpha', np.linspace(-5., 12., nn), units='deg')

prob.model.add_subsystem('SimpleWing', SimpleWing(num_nodes=nn), promotes=['*'])

prob.setup(check=False, force_alloc_complex=True)

prob.run_model()


import matplotlib.pyplot as plt

drag = prob['drag'] / 1000.
lift = prob['lift'] / 1000.

plt.figure(figsize=(10, 8))

plt.plot(drag, lift)

indices = [0, nn//2, -1]

x_scatter = drag[indices]
y_scatter = lift[indices]

plt.scatter(x_scatter, y_scatter, color='k', zorder=10)

plt.annotate('Alpha = {:.1f} deg'.format(prob['alpha'][indices[0]]), xy=(x_scatter[0]+10., y_scatter[0]))
plt.annotate('Alpha = {:.1f} deg'.format(prob['alpha'][indices[1]]), xy=(x_scatter[1]+15., y_scatter[1]))
plt.annotate('Alpha = {:.1f} deg'.format(prob['alpha'][indices[2]]), xy=(x_scatter[2]-130., y_scatter[2]-200.))


plt.xlabel('Drag, kN')
plt.ylabel('Lift, kN')

plt.tight_layout()
plt.savefig('drag_polar.pdf')
