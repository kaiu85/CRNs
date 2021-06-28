import pickle

# FOR CUDA DEBUGGING, c.f. https://lernapparat.de/debug-device-assert/
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "0"# Set to "1" for more verbose debugging messages

import torch
from network_constructors import create_wta_model

from utils import save_trajectories
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Select GPU
torch.cuda.set_device(0)

print_debug = False

steps = 300000

log_every = 100
n = 1000
cont = False

forcings = [100, 500]

max_plots = 10

cm = 1/2.54  # centimeters in inches

### Simulation

for forcing in forcings:
    
    suffix = 'wta_demo_network_' + str(forcing)
    
    CRN = create_wta_model(mean_forcing = forcing)
    
    CRN.A[:,:] = torch.randint(0,500,CRN.A.shape)
    
    CRN.init_global_reaction_variables(print_debug = print_debug)

    print('Simulating network consisting of %d species and %d/2 reactions!' % (CRN.N, CRN.M))

    with open( './results/network' + suffix + '.obj', 'wb') as network_file:
        pickle.dump(CRN, network_file)

    if cont:
        As = np.load('./results/As' + suffix + '.npy')
        CRN.A = torch.tensor(As[:,:,-1].squeeze()).cuda()
        ts = np.load('./results/ts' + suffix + '.npy')
        CRN.t = torch.tensor(ts[:,-1].squeeze()).cuda()

        suffix = suffix + '_cont'

    results = CRN.run(steps, log_every = log_every, print_debug = print_debug)
    save_trajectories(results, './results/', suffix)

    
### Plotting

for forcing in forcings:
    
    suffix = 'wta_demo_network_' + str(forcing)    
   
    ts = np.load('./results/ts' + suffix + '.npy')
    As = np.load('./results/As' + suffix + '.npy')

    for j in range(min(ts.shape[0],max_plots)):

        plt.figure(figsize = (30*cm, 20*cm))
        plt.plot(ts[j,:],As[:,j,:].squeeze().transpose(), label = None)

        plt.savefig("./figures/Fig_1b_WTA_forcing_%d_trajectory_id_%d.svg" % (forcing, j) )
    