import numpy as np
from tqdm import tqdm
import torch
from math import floor

class reaction:
    def __init__(self, N, educts, products, catalysts, ks):
        self.N = N        
        
        self.educts = educts
        self.educts.sort()
       
        self.products = products
        self.products.sort()
        if catalysts is not None:
            self.catalysts = catalysts
            self.catalysts.sort()
        else:
            self.catalysts = None
        
        self.subtract_educts = []
        self.subtract_catalysts = []
        
        for i in range(len(self.educts)):
            if i == 0:
                self.subtract_educts.append(0)
            else:
                if self.educts[i-1] == self.educts[i]:
                    self.subtract_educts.append( self.subtract_educts[-1] - 1 )
                else:
                    self.subtract_educts.append( 0 )
                    
        self.subtract_educts = torch.reshape(torch.tensor(self.subtract_educts),(-1,1)).cuda()
        
        if self.catalysts is not None:
            for i in range(len(self.catalysts)):
                if i == 0:
                    self.subtract_catalysts.append( -self.educts.count( self.catalysts[i] ) )
                else:
                    if self.catalysts[i-1] == self.catalysts[i]:
                        self.subtract_catalysts.append( self.subtract_catalysts[-1] - 1 )
                    else:
                        self.subtract_catalysts.append( -self.educts.count( self.catalysts[i] ) )
                        
            self.subtract_catalysts = torch.reshape(torch.tensor(self.subtract_catalysts),(-1,1)).cuda()
        else:
            self.subtract_catalysts = None
        
        self.k = torch.reshape(torch.tensor(np.random.choice(ks,1)), (1,)).cuda()
        
    def get_propensity(self, A):
        
        if self.catalysts is None:
            prop = self.k*(A[self.educts,:] + self.subtract_educts).prod(axis = 0)
        else:
            prop = self.k*(A[self.educts,:] + self.subtract_educts).prod(axis = 0)*(A[self.catalysts,:] - self.subtract_catalysts).prod(axis = 0)
        
        # Make sure propensities are always positive
        prop = torch.max(prop, 0.0*prop)
        
        return prop
    
    def get_difference_vector(self):
        diff = torch.zeros((self.N,)).cuda()
        
        for i in range(len(self.educts)):
            diff[self.educts[i]] -= 1.0
            
        for i in range(len(self.products)):
            diff[self.products[i]] += 1.0
            
        return diff        
 
        
class stochastic_reaction_network:
    # initializers
    def __init__(self, N = 25, n = 100, p_bimolecular = 0.3, p_catalytic = 0.5, steps = int(1e6), ks = [1, 1e-1, 1e-2, 1e-3], create_reactions = True, M = 35, local_detailed_balance = None, equilibrium_potentials = None, dt = 1e-3):
        
        self.N = N
        
        self.n = n
        
        self.A = torch.zeros( N, n ).cuda()
        self.t = torch.zeros( n ).cuda()
        
        self.dt = dt
        self.clamp_times = []
        
        self.clamped = dict()
        self.clamp_states = dict()
        
        self.reactions = []
        
        if create_reactions:
            
            self.M = M
            
            self.P = torch.zeros( M, n ).cuda()
            self.D = torch.zeros( N, M ).cuda()
            
            # Create baseline reactions
            for i in range(M):

                # Create monomolecular reaction
                if np.random.rand() < 1.0 - p_bimolecular:
                    chemicals = np.random.choice(N,2,replace = False)
                    educt = [chemicals[0]]
                    product = [chemicals[1]]

                    self.reactions.append(reaction(N, educt, product, None, ks))

                # Create non-catalytic bimolecular reaction
                elif np.random.rand() < 1.0 - p_catalytic:
                    chemicals = np.random.choice(N,4,replace = False)
                    educt = chemicals[0:2].tolist()
                    product = chemicals[2:4].tolist()

                    self.reactions.append(reaction(N, educt, product, None, ks))

                # Create catalytic bimolecular reaction
                else:
                    chemicals = np.random.choice(N,5,replace = False)
                    educt = chemicals[0:2].tolist()
                    product = chemicals[2:4].tolist()
                    catalyst = [chemicals[4]]

                    self.reactions.append(reaction(N, educt, product, catalyst, ks))
                
                diff = self.reactions[-1].get_difference_vector()
                
                self.D[:,i] = diff
                                    
    def clamp_species(self, species, clamp_function, clamp_state = None):
        
        self.clamped[species] = clamp_function
        self.clamp_states[species] = clamp_state
        
    def set_counts(self, A):
        
        if len(A.shape) == 1:
            if self.A.shape[0] == A.shape[0]:
                A = A.unsqueeze(1)
                self.A = A.repeat([1, self.n])
            else:
                print('ERROR! Old and new shapes of A incompatible!')
                print(self.A.shape)
                print('VS.:')
                print(A.shape)
        elif len(A.shape) == 2:        
            if self.A.shape[0] == A.shape[0] and self.A.shape[1] == A.shape[1]:
                self.A = A
            else:
                print('ERROR! Old and new shapes of A incompatible!')
                print(self.A.shape)
                print('VS.:')
                print(A.shape)
        else:
            print('ERROR! Old and new shapes of A incompatible!')
            print(self.A.shape)
            print('VS.:')
            print(A.shape)
        
    def set_clamp_times(self, clamp_times):
        
        clamp_times = np.sort(clamp_times)
        self.clamp_times = torch.tensor(clamp_times).type(torch.FloatTensor).cuda()
        
    def get_potential(self, species):
        
        pot = np.log(self.A[species])
        
        return pot
    
    def get_potentials(self):
        
        pot = np.log(self.A)
        
        return pot

    def init_global_reaction_variables(self, print_debug = False):
        
        self.M = len(self.reactions)
        
        self.P = torch.zeros( self.M, self.n ).cuda()
        
        self.D = torch.zeros( self.N, self.M ).cuda()

        for i in range(self.M):
            
            diff = self.reactions[i].get_difference_vector()
            self.D[:,i] = diff
            
            
        self.ME1 = torch.zeros( self.M, self.N ).cuda()
        self.ME2 = torch.zeros( self.M, self.N ).cuda()
        self.VE1 = torch.ones( self.M, 1).cuda()
        self.VE2 = torch.ones( self.M, 1).cuda()
        
        self.MC1 = torch.zeros( self.M, self.N ).cuda()
        self.MC2 = torch.zeros( self.M, self.N ).cuda()
        self.VC1 = torch.ones( self.M, 1).cuda()
        self.VC2 = torch.ones( self.M, 1).cuda()
        
        self.k = torch.zeros( self.M, 1).cuda()
        
        for i in range(len(self.reactions)):
            
            self.k[i] = self.reactions[i].k
            
            self.ME1[i,self.reactions[i].educts[0]] = 1.0
            self.VE1[i] = self.reactions[i].subtract_educts[0]
            
            if len(self.reactions[i].educts) == 2:
                
                self.ME2[i,self.reactions[i].educts[1]] = 1.0
                self.VE2[i] = self.reactions[i].subtract_educts[1]
                
            if self.reactions[i].catalysts is not None:
                self.MC1[i,self.reactions[i].catalysts[0]] = 1.0
                self.VC1[i] = self.reactions[i].subtract_catalysts[0]
                    
                if len(self.reactions[i].catalysts) == 2:
                    self.MC2[i,self.reactions[i].catalysts[1]] = 1.0
                    self.VC2[i] = self.reactions[i].subtract_catalysts[1]
            
            if print_debug: 
                print('Reaction %d:\n' % i)
                print('Products:')
                print(self.reactions[i].products)
                print('Educts:')
                print(self.reactions[i].educts)
                print(self.reactions[i].subtract_educts)
                print('Catalysts:')
                print(self.reactions[i].catalysts)
                print(self.reactions[i].subtract_catalysts)
            
        if print_debug:        
            print('Stochiometric Matrices:\n')

            print(self.ME1)
            print(self.VE1)

            print(self.ME2)
            print(self.VE2)

            print(self.MC1)
            print(self.VC1)

            print(self.MC2)
            print(self.VC2)

            print(self.k)
                
    def get_propensities(self, A, print_debug = False):
        
        #for i in range(len(self.reactions)):
        #    self.P[i,:] = self.reactions[i].get_propensity(A)
        
        E1 = torch.matmul(self.ME1, A) + self.VE1
        E2 = torch.matmul(self.ME2, A) + self.VE2
        C1 = torch.matmul(self.MC1, A) + self.VC1
        C2 = torch.matmul(self.MC2, A) + self.VC2
        
        if print_debug:
            print('A, E1, E2, C1, C2:')
            print(A)
            print(E1)
            print(E2)
            print(C1)
            print(C2)
        
        self.P = self.k*E1*E2*C1*C2
     
    def get_segment_start_times(self, segment_indices):
        
        if len(self.clamp_times) == 0 and self.dt is not None:
            segment_start_times = segment_indices*self.dt
        elif len(self.clamp_times) > 0:
            n_boundaries = len(self.clamp_times)
            n_segments = n_boundaries - 1
            
            within_boundaries = torch.where(segment_indices < (n_segments))
            outside_boundaries = torch.where(segment_indices >= (n_segments))
            
            segment_start_times = 0.0*segment_indices
            
            segment_start_times[within_boundaries] = self.clamp_times[segment_indices[within_boundaries]]
            segment_start_times[outside_boundaries] = float('inf')
            
        return segment_start_times
                    
    def get_segment_end_times(self, segment_indices):
        return self.get_segment_start_times(segment_indices + 1)
        
    def get_segment_indices(self, times):
        if len(self.clamp_times) == 0 and self.dt is not None:
            segment_indices = torch.floor(times/self.dt).type(torch.LongTensor)
        elif len(self.clamp_times) > 0:
            n_boundaries = len(self.clamp_times)
            n_segments = n_boundaries - 1
            
            segment_indices = (0.0*times + n_segments).type(torch.LongTensor)
            
            for i in range(n_segments):
                hits = torch.where( ( times >= self.clamp_times[i] ) & ( times < self.clamp_times[i+1] ) )[0]
                segment_indices[hits] = i
                                   
        return segment_indices.cuda()
        
    # forward method
    def run(self, steps = 100000,
            log = True, log_every = 1, print_debug = False, eps = 1e-12):
        
        if log:
            
            ts = torch.zeros(self.n, int(floor(steps/log_every)))
            Rs = torch.zeros(self.n, int(floor(steps/log_every)))

            updates = torch.zeros(self.n, int(floor(steps/log_every)))

            As = torch.zeros(self.N, self.n, int(floor(steps/log_every)))
            Ps = torch.zeros(self.M, self.n, int(floor(steps/log_every)))
            
            reactions_triggered = -1*torch.ones(self.n, int(floor(steps/log_every))).long()
        
        # indexes the time-segment w.r.t. time-dependent driving forces
        segment_indices = self.get_segment_indices(self.t)
        
        # iterate over given number of steps
        for i in tqdm(range(steps)):
            
            triggered = 0
            
            for species, clamp_function in self.clamped.items():
                start_times = self.get_segment_start_times(segment_indices)
                self.A[species,:], self.clamp_states[species] = clamp_function(start_times, segment_indices, self.clamp_states[species])
                
            if log and (i % log_every) == 0:
                
                logi = int(floor(i/log_every))
                
                ts[:,logi] = self.t.cpu()
                As[:,:,logi] = self.A.cpu()
            
            if i == 0:
                print_propensities = print_debug
            else:
                print_propensities = False
                
            self.get_propensities(self.A, print_propensities)
            
            self.R = self.P.sum(axis = 0)
            
            # If the sum of all rates is 0, do nothing EXCEPT MOVING TIME FORWARD!
            update_indices = torch.where(self.R > eps)[0]
            move_time_indices = torch.where(self.R <= eps)[0]
            
            if len(move_time_indices) > 0:
                
                #print('\nMoving time for processes:')
                #print(move_time_indices)
                
                #print('Before:')
                #print(self.t[move_time_indices])
                
                self.t[move_time_indices] = self.get_segment_end_times(segment_indices[move_time_indices])
                segment_indices[move_time_indices] += 1
                
                #print('After:')
                #print(self.t[move_time_indices])
                
                #print('Rs:')
                #print(self.R[move_time_indices])
            
            if len(update_indices) > 0:
                
                # Double precision is important here, since for float tensors due to the
                # discretization "0" would occur way to often (and would get mapped to delta_t = inf)
                noise = torch.cuda.DoubleTensor(len(update_indices)).uniform_()
                delta_t =  - torch.log( noise ) / self.R[update_indices]
                
                skip_indices = torch.where( 
                                self.t[update_indices] + delta_t >
                                self.get_segment_end_times(segment_indices[update_indices])
                               )[0]
                do_indices = torch.where( 
                              self.t[update_indices] + delta_t <= 
                              self.get_segment_end_times(segment_indices[update_indices])
                             )[0]
                
                if len(skip_indices) > 0:
                    
                    '''
                    if delta_t[skip_indices].max().cpu().item() > 100.0:
                    
                        print('\nSkipping time for processes:')
                        print(update_indices[skip_indices])

                        print('Before:')
                        print(self.t[update_indices[skip_indices]])
                    '''

                    self.t[update_indices[skip_indices]] = \
                        self.get_segment_end_times(segment_indices[update_indices[skip_indices]])
                    segment_indices[update_indices[skip_indices]] += 1

                    '''
                    if delta_t[skip_indices].max().cpu().item() > 100.0:
                        
                        print('After:')
                        print(self.t[update_indices[skip_indices]])

                        print('delta_ts:')
                        print(delta_t[skip_indices])
                        
                        print('Rs:')
                        print(self.R[update_indices[skip_indices]])
                              
                        print('noise:')
                        print(noise[skip_indices])
                    '''
                    
                if len(do_indices) > 0:
                    
                    triggered = 1
                    
                    self.t[update_indices[do_indices]] += delta_t[do_indices]
                    
                    self.p = self.P[:,update_indices[do_indices]] / self.R[update_indices[do_indices]]
                    Cat = torch.distributions.categorical.Categorical(probs=torch.transpose(self.p,1,0))
                    reactions = Cat.sample()

                    self.A[:,update_indices[do_indices]] += self.D[:,reactions]

            if log and (i % log_every) == 0:
                
                logi = int(floor(i/log_every))
                
                updates[update_indices,logi] = 1.0

                Rs[:,logi] = self.R.cpu()
                Ps[:,:,logi] = self.P.cpu()
                
                if triggered == 1:
                    reactions_triggered[update_indices[do_indices],logi] = reactions.cpu()
                
        if log:
            return [ts, As, updates, Rs, Ps, reactions_triggered]
        else:
            return
