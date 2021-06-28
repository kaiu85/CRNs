from matplotlib import pyplot as plt
from simple_driven_network import simple_reaction_network, reaction
import numpy as np

dt = 0.01
steps = 100000

N = 9

plot = True

forcings = [500.0, 600.0, 700.0, 750.0, 800.0, 850.0, 900.0, 950.0, 1000.0, 1050.0, 1100.0, 1150.0, 1200.0, 1250.0, 1300.0, 1350.0, 1400.0, 1450.0, 1500.0, 2000.0, 2500.0, 3000.0]

i = 0

efficiencies = np.zeros( (len(forcings),3) )

#### Simulate deterministic CRNs and calculate efficiencies

for i, forcing in enumerate(forcings):
    
    print('Calculating efficiencies for forcing %f' % forcing)

    for state in range(3):

        CRN = simple_reaction_network(N = N, M = 35, steps = int(1e6), ks = [], dt = 0.01, create_reactions = False)

        # Intermediates for different relaxation channels

        # Channel 1
        CRN.reactions.append( reaction(N, [0], [6], [1,1], [1e-5]) )
        CRN.reactions.append( reaction(N, [4], [6], None, [1.0]) )

        # Channel 2
        CRN.reactions.append( reaction(N, [0], [7], [2,2], [1e-5]) )
        CRN.reactions.append( reaction(N, [4], [7], None, [1.0]) )

        # Channel 3
        CRN.reactions.append( reaction(N, [0], [8], [3,3], [1e-5]) )
        CRN.reactions.append( reaction(N, [4], [8], None, [1.0]) )

        # Reservoir for chemical switches
        CRN.reactions.append( reaction(N, [6], [5], [1,1], [1e-6]) )
        CRN.reactions.append( reaction(N, [7], [5], [2,2], [3e-7]) )
        CRN.reactions.append( reaction(N, [8], [5], [3,3], [1e-7]) )

        # Chemical switch 1
        CRN.reactions.append( reaction(N, [5], [1], [1,1], [1e-6]) )
        CRN.reactions.append( reaction(N, [1], [4], [2,3], [0.001]) )

        # Chemical switch 2
        CRN.reactions.append( reaction(N, [5], [2], [2,2], [1e-6]) )
        CRN.reactions.append( reaction(N, [2], [4], [1,3], [0.001]) )

        # Chemical switch 3
        CRN.reactions.append( reaction(N, [5], [3], [3,3], [1e-6]) )
        CRN.reactions.append( reaction(N, [3], [4], [1,2], [0.001]) )

        if state == 0:
            CRN.A[1] = forcing
            CRN.A[2] = 5.0
            CRN.A[3] = 5.0
            CRN.A[5] = forcing
            CRN.A[6] = forcing
            CRN.A[7] = 5.0
            CRN.A[8] = 5.0
        elif state == 1:
            CRN.A[1] = 5.0
            CRN.A[2] = forcing
            CRN.A[3] = 5.0
            CRN.A[5] = forcing
            CRN.A[6] = 5.0
            CRN.A[7] = forcing
            CRN.A[8] = 5.0
        else:
            CRN.A[1] = 5.0
            CRN.A[2] = 5.0
            CRN.A[3] = forcing
            CRN.A[5] = forcing
            CRN.A[6] = 5.0
            CRN.A[7] = 5.0
            CRN.A[8] = forcing        

        # Clamp concentration of species 4 to low concentration
        CRN.clamp_concentration(4, 5.0)

        # Clamp concentration of species 0 to high concentration
        CRN.clamp_concentration(0, forcing)

        A, fluxes, delta_Gs, clamped_fluxes, clamped_potentials, potentials, AG = CRN.run(steps, log_fluxes = True)
       
        eprs = -(fluxes*delta_Gs)

        epr_total = eprs.sum(axis = 1)
       
        print('Efficiency of state %d:' % state)
        efficiency = (1.0 - (eprs[-1,0] + eprs[-1,1] + eprs[-1,2] + eprs[-1,3] + eprs[-1,4] + eprs[-1,5])/epr_total[-1])
        print(efficiency)
        stable = A[-1, state + 1] > 0.5*forcing
        print('Stable:')
        print(stable)
        
        if stable:
            efficiencies[i,state] = efficiency
        else:
            efficiencies[i,state] = 0.0
            
np.save('./results/efficiencies.npy', efficiencies)
np.save('./results/forcings_for_efficiencies.npy',np.array(forcings))

### Plot graph for figure 4b

efficiencies = np.load('./results/efficiencies.npy')
forcings = np.load('./results/forcings_for_efficiencies.npy')

colors = ['tab:orange','tab:green','tab:red']

cm = 1/2.54  # centimeters in inches
plt.figure(figsize = (30*cm, 20*cm))

for i in range(3):
    eff = efficiencies[:,i]
    idx = np.where( eff > 1e-12 )[0]
    f = forcings[idx]
    e = eff[idx]    
    plt.plot(f, e, color = colors[i], lw = 2.0)

plt.xlim([500, 3000])
    
plt.savefig("./figures/Fig_4b_efficiency_plot.svg")