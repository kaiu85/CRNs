import torch
import matplotlib.pyplot as plt
from network_constructors import create_turnstile_wta_model
from utils import save_trajectories
import numpy as np
    
steps = 6000000
log_every = 1000
print_debug = False
n = 4000

forcings = [500.0, 600.0, 700.0, 750.0, 800.0, 850.0, 900.0, 950.0, 1000.0, 1050.0, 1100.0, 1150.0, 1200.0, 1250.0, 1300.0, 1350.0, 1400.0, 1450.0, 1500.0, 2000.0, 2500.0, 3000.0]

# Forcings and trajectories to plot
plot_forcings = [1200.0]
trajectory_ids = [0, 1000, 2000, 3000]

cm = 1/2.54  # centimeters in inches
threshold = 0.5 # threshold to determine, which population won

###### Simulate processes

for i, forcing in enumerate(forcings):    
    
    print('Running simulation for forcing ' + str(forcing))

    suffix = 'wta_efficiency_final_network_new_spacing_' + str(forcing)

    CRN = create_turnstile_wta_model(mean_forcing = forcing, n = n)

    CRN.A[:,:] = torch.randint(0,int(forcing),CRN.A.shape)
                     
    CRN.A[5,:] = forcing
    
    CRN.A[1,:1000] = forcing
    CRN.A[2,:1000] = 5.0
    CRN.A[3,:1000] = 5.0
                
    CRN.A[6,:1000] = forcing
    CRN.A[7,:1000] = 5.0
    CRN.A[8,:1000] = 5.0
        
    CRN.A[1,1000:2000] = 5.0
    CRN.A[2,1000:2000] = forcing
    CRN.A[3,1000:2000] = 5.0
                
    CRN.A[6,1000:2000] = 5.0
    CRN.A[7,1000:2000] = forcing
    CRN.A[8,1000:2000] = 5.0
    
    CRN.A[1,2000:3000] = 5.0
    CRN.A[2,2000:3000] = 5.0
    CRN.A[3,2000:3000] = forcing
                
    CRN.A[6,2000:] = 5.0
    CRN.A[7,2000:] = 5.0
    CRN.A[8,2000:] = forcing
    
    CRN.A[1,3000:] = 5.0
    CRN.A[2,3000:] = 5.0
    CRN.A[3,3000:] = 5.0
    
    CRN.A[6,3000:] = 5.0
    CRN.A[7,3000:] = 5.0
    CRN.A[8,3000:] = 5.0

    CRN.init_global_reaction_variables(print_debug = print_debug)

    results = CRN.run(steps, log_every = log_every, print_debug = print_debug)
    save_trajectories(results, './results/', suffix)

###### Create graphs for figures 4a and 4c

ps = np.zeros( (len(forcings), 4) )
As_final = np.zeros( (9, n, len(forcings)) )

for i, forcing in enumerate(forcings):        

    suffix = 'wta_efficiency_final_network_new_spacing_' + str(forcing)
   
    ts = np.load('./results/ts' + suffix + '.npy')
    As = np.load('./results/As' + suffix + '.npy')
    
    legend_labels = []
    for j in range(As.shape[1]):
        legend_labels.append('Species ' + str(j))
    
    if forcing in forcings:
        for trajectory_id in trajectory_ids:
            if (forcing in plot_forcings):
                plt.figure(figsize = (30*cm, 20*cm))
                plt.plot(ts[trajectory_id,:],As[:,trajectory_id,:].transpose())
                plt.savefig("./figures/Fig_4c_efficiency_forcing_%d_trajectory_id_%d.svg" % (forcing, trajectory_id) )
        
    As_final[:,:,i] = As[:,:,-1]
        
    p1_wins = np.logical_and( As[1,:1000,-1] > threshold*forcing, np.logical_and( As[2,:1000,-1] <= threshold*forcing, As[3,:1000,-1] <= threshold*forcing))
    p1_wins = p1_wins.sum() / 1000
    
    p2_wins = np.logical_and( As[1,1000:2000,-1] <= threshold*forcing, np.logical_and( As[2,1000:2000,-1] > threshold*forcing, As[3,1000:2000,-1] <= threshold*forcing))
    p2_wins = p2_wins.sum() / 1000
    
    p3_wins = np.logical_and( As[1,2000:3000,-1] <= threshold*forcing, np.logical_and( As[2,2000:3000,-1] <= threshold*forcing, As[3,2000:3000,-1] > threshold*forcing))
    p3_wins = p3_wins.sum() / 1000
    
    s_wins = np.logical_and( As[1,3000:,-1] <= threshold*forcing, np.logical_and( As[2,3000:,-1] <= threshold*forcing, As[3,3000:,-1] <= threshold*forcing))
    s_wins = s_wins.sum() / 1000
    
    ps[i,0] = p1_wins
    ps[i,1] = p2_wins
    ps[i,2] = p3_wins
    ps[i,3] = s_wins
    
cm = 1/2.54  # centimeters in inches
plt.figure(figsize = (30*cm, 20*cm))
plt.plot(forcings, ps[:,[3,0,1,2]], lw = 2.0)
plt.xlim([500.0, 3000.0])
plt.savefig("./figures/Fig_4a.svg")