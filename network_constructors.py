from tssa import stochastic_reaction_network, reaction
import torch
import numpy as np
from functools import partial
  
def clamp_fixed_flex(ts, segment_indices, clamp_state, mean_forcing = 1000.0):
    
    species = 0*ts + mean_forcing
    
    return species, clamp_state    

def create_turnstile_wta_model(mean_forcing = 1000, n = 10, dt = 0.01):
    
    clamp_high_forcing = partial(clamp_fixed_flex, mean_forcing = mean_forcing)
    clamp_low_forcing = partial(clamp_fixed_flex, mean_forcing = 5)
    
    N = 9
    
    CRN = stochastic_reaction_network(N = N, n = n, dt = dt, create_reactions = False)
    
    # Intermediates for different relaxation channels
    
    # Channel 1
    CRN.reactions.append( reaction(N, [0], [6], [1,1], [1e-5]) )
    CRN.reactions.append( reaction(N, [6], [0], [1,1], [1e-5]) )
    
    CRN.reactions.append( reaction(N, [4], [6], None, [1.0]) )
    CRN.reactions.append( reaction(N, [6], [4], None, [1.0]) )
    
    # Channel 2
    CRN.reactions.append( reaction(N, [0], [7], [2,2], [1e-5]) )
    CRN.reactions.append( reaction(N, [7], [0], [2,2], [1e-5]) )
    
    CRN.reactions.append( reaction(N, [4], [7], None, [1.0]) )
    CRN.reactions.append( reaction(N, [7], [4], None, [1.0]) )
    
    # Channel 3
    CRN.reactions.append( reaction(N, [0], [8], [3,3], [1e-5]) )
    CRN.reactions.append( reaction(N, [8], [0], [3,3], [1e-5]) )
    
    CRN.reactions.append( reaction(N, [4], [8], None, [1.0]) )
    CRN.reactions.append( reaction(N, [8], [4], None, [1.0]) )
    
    # Reservoir for chemical switches
    CRN.reactions.append( reaction(N, [6], [5], [1,1], [1e-6]) )
    CRN.reactions.append( reaction(N, [5], [6], [1,1], [1e-6]) )
    
    CRN.reactions.append( reaction(N, [7], [5], [2,2], [3e-7]) )
    CRN.reactions.append( reaction(N, [5], [7], [2,2], [3e-7]) )
    
    CRN.reactions.append( reaction(N, [8], [5], [3,3], [1e-7]) )
    CRN.reactions.append( reaction(N, [5], [8], [3,3], [1e-7]) )
    
    # Chemical switch 1
    CRN.reactions.append( reaction(N, [5], [1], [1,1], [1e-6]) )
    CRN.reactions.append( reaction(N, [1], [5], [1,1], [1e-6]) )
    
    CRN.reactions.append( reaction(N, [1], [4], [2,3], [0.001]) )
    CRN.reactions.append( reaction(N, [4], [1], [2,3], [0.001]) )
    
    # Chemical switch 2
    CRN.reactions.append( reaction(N, [5], [2], [2,2], [1e-6]) )
    CRN.reactions.append( reaction(N, [2], [5], [2,2], [1e-6]) )
    
    CRN.reactions.append( reaction(N, [2], [4], [1,3], [0.001]) )
    CRN.reactions.append( reaction(N, [4], [2], [1,3], [0.001]) )
    
    # Chemical switch 3
    CRN.reactions.append( reaction(N, [5], [3], [3,3], [1e-6]) )
    CRN.reactions.append( reaction(N, [3], [5], [3,3], [1e-6]) )
    
    CRN.reactions.append( reaction(N, [3], [4], [1,2], [0.001]) )
    CRN.reactions.append( reaction(N, [4], [3], [1,2], [0.001]) )
    
    # Clamp concentration of species 4 to low concentration
    CRN.clamp_species(4, clamp_low_forcing)
    
    # Clamp concentration of species 0 to high concentration
    CRN.clamp_species(0, clamp_high_forcing)
    
    # Initialize all species at zero number counts
    CRN.A[:,:] = 0
    
    return CRN

def create_wta_model(mean_forcing = 1000, n = 10, dt = 0.01):
    
    clamp_high_forcing = partial(clamp_fixed_flex, mean_forcing = mean_forcing)
    clamp_low_forcing = partial(clamp_fixed_flex, mean_forcing = 5)
    
    N = 5
    
    CRN = stochastic_reaction_network(N = N, n = n, dt = dt, create_reactions = False)
    
    # Chemical switch 1
    CRN.reactions.append( reaction(N, [0], [1], [1,1], [1e-6]) )
    CRN.reactions.append( reaction(N, [1], [0], [1,1], [1e-6]) )
    
    CRN.reactions.append( reaction(N, [1], [4], [2,3], [0.001]) )
    CRN.reactions.append( reaction(N, [4], [1], [2,3], [0.001]) )
    
    # Chemical switch 2
    CRN.reactions.append( reaction(N, [0], [2], [2,2], [1e-6]) )
    CRN.reactions.append( reaction(N, [2], [0], [2,2], [1e-6]) )
    
    CRN.reactions.append( reaction(N, [2], [4], [1,3], [0.001]) )
    CRN.reactions.append( reaction(N, [4], [2], [1,3], [0.001]) )
    
    # Chemical switch 3
    CRN.reactions.append( reaction(N, [0], [3], [3,3], [1e-6]) )
    CRN.reactions.append( reaction(N, [3], [0], [3,3], [1e-6]) )
    
    CRN.reactions.append( reaction(N, [3], [4], [1,2], [0.001]) )
    CRN.reactions.append( reaction(N, [4], [3], [1,2], [0.001]) )
    
    # Clamp concentration of species 4 to low concentration
    CRN.clamp_species(4, clamp_low_forcing)
    
    # Clamp concentration of species 0 to high concentration
    CRN.clamp_species(0, clamp_high_forcing)
    
    # Initialize all species at zero number counts
    CRN.A[:,:] = 0
    
    return CRN




