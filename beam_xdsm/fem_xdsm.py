from pyxdsm.XDSM import XDSM

#
opt = 'Optimization'
solver = 'MDA'
ecomp = 'Analysis'
icomp = 'ImplicitAnalysis'


x = XDSM()

x.add_system('moi', ecomp, [r'\text{1) moment of inertia}',
                            r'I_i = \frac{1}{4} b * h_i^3'])

x.add_system('local_K', ecomp, [r'\text{2) local stiffness:}',
                                r'\left[K_\text{local}\right]_i'])

x.add_system('global_K', icomp, [r'\text{3) FEM:}',
                                 r'\left[K_\text{global}\right] u = f', 
                                 r'u = [d, \theta]'])

x.add_system('compliance', ecomp, [r'\text{4) compliance:}', 
                                   r'c = f \cdot d'])

x.add_system('volume', ecomp, [r'\text{5) volume:}', 
                               r'\sum h_i b L_i'])

x.connect('moi', 'local_K', 'I')
x.connect('local_K', 'global_K', r'K_\text{local}')
x.connect('global_K', 'compliance', r'd')

x.add_input('volume', 'h')
x.add_input('moi', 'h')

x.write('fem_xdsm')
