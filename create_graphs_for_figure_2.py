import torch
import matplotlib.pyplot as plt
from network_constructors import create_wta_model
from utils import save_trajectories
import numpy as np

# C.f. Rao & Esposito (2018), equation (42)
def calculate_gibbs_free_energy(As):
    gs = torch.lgamma(As + 1.0)
    gs = gs.sum(dim = [0])
    
    return gs
    
steps = 300000
log_every = 3000
print_debug = False
n = 100000

# Number of initial reaction events to record, after system is closed
n_dir = 10

steps_eval = n_dir + 1
log_every_eval = 1

# List of forcings to simulate and analyze
forcings =  list(np.arange(100.0, 400.0, 100.0)) + list(np.arange(310.0, 400.0, 10.0)) + list(np.arange(400.0, 2400.0, 100.0))

print('Number of forcings to simulate: ' + str(len(forcings)))

plot_forcings = [200.0, 400.0]

max_plots = 10

cm = 1/2.54  # centimeters in inches

####### SIMULATE CONVERGENCE TO STEADY STATE AND INITIAL PHASE OF DECAY

for forcing in forcings:   

    print('Simulating CRN initialized at high-concentration state for forcing ' + str(forcing) )
    
    suffix = 'wta_energy_calculation_network_final_' + str(forcing)

    CRN = create_wta_model(mean_forcing = forcing, n = n)

    CRN.A[:,:] = torch.randint(0,500,CRN.A.shape)

    CRN.A[1,:] = forcing
    CRN.A[2,:] = 5.0
    CRN.A[3,:] = 5.0

    CRN.init_global_reaction_variables(print_debug = print_debug)

    results = CRN.run(steps, log_every = log_every, print_debug = print_debug)
    save_trajectories(results, './results/', suffix)
    
    ts = np.load('./results/ts' + suffix + '.npy')
    As = np.load('./results/As' + suffix + '.npy')   
   
    suffix = 'decaying_' + suffix
    
    CRN2 = create_wta_model(mean_forcing = forcing, n = n)

    CRN2.clamped = dict()

    CRN2.A = torch.tensor(As[:,:,-1]).cuda()

    CRN2.init_global_reaction_variables(print_debug = print_debug)

    results = CRN2.run(steps_eval, log_every = log_every_eval, print_debug = print_debug)
    save_trajectories(results, './results/', suffix)   
    
    
for forcing in forcings:   

    print('Simulating CRN initialized at low-concentration state for forcing ' + str(forcing) )
    
    suffix = 'wta_energy_calculation_network_final_low' + str(forcing)

    CRN = create_wta_model(mean_forcing = forcing, n = n)

    CRN.A[:,:] = torch.randint(0,500,CRN.A.shape)
   
    CRN.A[1,:] = 5.0
    CRN.A[2,:] = 5.0
    CRN.A[3,:] = 5.0

    CRN.init_global_reaction_variables(print_debug = print_debug)

    results = CRN.run(steps, log_every = log_every, print_debug = print_debug)
    save_trajectories(results, './results/', suffix)
    
    ts = np.load('./results/ts' + suffix + '.npy')
    As = np.load('./results/As' + suffix + '.npy')
    
    suffix = 'decaying_' + suffix
    
    CRN2 = create_wta_model(mean_forcing = forcing, n = n)

    CRN2.clamped = dict()

    CRN2.A = torch.tensor(As[:,:,-1]).cuda()

    CRN2.init_global_reaction_variables(print_debug = print_debug)

    results = CRN2.run(steps_eval, log_every = log_every_eval, print_debug = print_debug)
    save_trajectories(results, './results/', suffix)    

##### CREATE GRAPHS FOR FIGURE 2

Wmins_direct_high = []
Wmins_std_high = []
forcings_high = []

Wmins_direct_low = []
Wmins_std_low = []
forcings_low = []

for forcing in forcings:   

    suffix = 'wta_energy_calculation_network_final_' + str(forcing)    
    
    if forcing in plot_forcings:
        
        ts = np.load('./results/ts' + suffix + '.npy')
        As = np.load('./results/As' + suffix + '.npy')
        
        for j in range(min(ts.shape[0],max_plots)):

            plt.figure(figsize = (30*cm, 20*cm))
            plt.plot(ts[j,:],As[:,j,:].squeeze().transpose(), label = None)            

            plt.savefig("./figures/Fig_2cd_WTA_%d_%d.svg" % (forcing, j) )    
    
    
    suffix = 'decaying_' + suffix
    
    ts = np.load('./results/ts' + suffix + '.npy')
    As = np.load('./results/As' + suffix + '.npy')

    gs = calculate_gibbs_free_energy(torch.tensor(As)).numpy()      
    
    high_indices = ( As[1,:,-1] > 0.5*forcing )
    
    print('%d of %d trajectories stayed in high-concentration state' % (high_indices.sum(), n) )
    
    # Only calculate heat production rate, if at least 1000 of the trajectories
    # stayed in a high concentration state
    if high_indices.sum() > 1000:
        Wmins_direct_high.append( - np.divide(gs[:,n_dir]-gs[:,0], ts[:,n_dir]-ts[:,0])[high_indices].mean() )
        Wmins_std_high.append( np.divide(gs[:,n_dir]-gs[:,0], ts[:,n_dir]-ts[:,0])[high_indices].std()/np.sqrt(len(high_indices)) )
        forcings_high.append(forcing)       
    
for forcing in forcings:   

    suffix = 'decaying_wta_energy_calculation_network_final_low' + str(forcing)   
    
    ts = np.load('./results/ts' + suffix + '.npy')
    As = np.load('./results/As' + suffix + '.npy')

    gs = calculate_gibbs_free_energy(torch.tensor(As)).numpy()
    
    low_indices = np.logical_and( As[1,:,-1] < 0.5*forcing , np.logical_and( As[2,:,-1] < 0.5*forcing, As[3,:,-1] < 0.5*forcing ) )
    
    print('%d of %d trajectories stayed in spontaneous state' % (low_indices.sum(), n) )    
        
    # Only calculate heat production rate, if at least 1000 of the trajectories
    # stayed in a low concentration state    
    if low_indices.sum() > 1000:
        Wmins_direct_low.append( - np.divide(gs[:,n_dir]-gs[:,0], ts[:,n_dir]-ts[:,0])[low_indices].mean() )
        Wmins_std_low.append( np.divide(gs[:,n_dir]-gs[:,0], ts[:,n_dir]-ts[:,0])[1 - high_indices].std()/np.sqrt(n-len(high_indices)) )
        forcings_low.append(forcing)
        

cm = 1/2.54  # centimeters in inches
plt.figure(figsize = (30*cm, 20*cm))

plt.figure()
plt.plot(forcings_low,Wmins_direct_low)
plt.fill_between(forcings_low, np.array(Wmins_direct_low) - np.array(Wmins_std_low), np.array(Wmins_direct_low) + np.array(Wmins_std_low), color = 'lightblue' )
plt.plot(forcings_high,Wmins_direct_high)
plt.fill_between(forcings_high, np.array(Wmins_direct_high) - np.array(Wmins_std_high), np.array(Wmins_direct_high) + np.array(Wmins_std_high), color = 'papayawhip' )
plt.xlim([100.0, 1400.0])
plt.ylim([-10.0, 200.0])
plt.savefig("./figures/Fig_2a.svg")

plt.figure()
plt.plot(forcings_low,Wmins_direct_low)
plt.fill_between(forcings_low, np.array(Wmins_direct_low) - np.array(Wmins_std_low), np.array(Wmins_direct_low) + np.array(Wmins_std_low), color = 'lightblue' )
plt.plot(forcings_high,Wmins_direct_high)
plt.fill_between(forcings_high, np.array(Wmins_direct_high) - np.array(Wmins_std_high), np.array(Wmins_direct_high) + np.array(Wmins_std_high), color = 'papayawhip' )
plt.xlim([200.0, 400.0])
plt.ylim([-10.0, 50.0])
plt.savefig("./figures/Fig_2b.svg")