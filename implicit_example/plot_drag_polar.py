from openmdao.api import Problem, Group, IndepVarComp

prob = Problem()
prob.model = Group()
des_vars = prob.model.add_subsystem('des_vars', IndepVarComp(), promotes=['*'])

nn = 51

des_vars.add_output('velocity', 200. * np.ones(nn), units='m/s')
des_vars.add_output('rho', 1.2 * np.ones(nn), units='kg/m**3')
des_vars.add_output('S_ref', 594720 * np.ones(nn), units='inch**2')
des_vars.add_output('alpha', np.linspace(-.2, 0.5, nn))

prob.model.add_subsystem('SimpleWing', SimpleWing(num_nodes=nn), promotes=['*'])

prob.setup(check=False, force_alloc_complex=True)

prob.run_model()

prob.check_partials(compact_print=True, method='fd')


prob.model.list_outputs(print_arrays=True)

import matplotlib.pyplot as plt

plt.plot(prob['drag'], prob['lift'])
plt.xlabel('drag')
plt.ylabel('lift')
plt.show()

plt.plot(prob['alpha'], prob['lift'])
plt.xlabel('alpha')
plt.ylabel('lift')
plt.show()

plt.plot(prob['alpha'], prob['drag'])
plt.xlabel('alpha')
plt.ylabel('drag')
plt.show()
