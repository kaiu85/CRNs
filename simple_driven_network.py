import numpy as np
from tqdm import tqdm

class reaction:
    def __init__(self, N, educts, products, catalysts, ks):
        self.N = N        
        
        self.educts = educts
        self.products = products
        self.catalysts = catalysts
        
        self.k = np.random.choice(ks,1)
        
    def get_flux(self, A):
        
        flux = np.zeros_like(A)
        
        if self.catalysts is None:
            Rplus = self.k*A[self.educts].prod()
            Rminus = self.k*A[self.products].prod()
        else:
            Rplus = self.k*A[self.educts].prod()*A[self.catalysts].prod()
            Rminus = self.k*A[self.products].prod()*A[self.catalysts].prod()
        
        flux[self.products] = Rplus - Rminus
        flux[self.educts] = Rminus - Rplus
        
        return flux
    
    def get_scalar_flux(self, A):
        
        if self.catalysts is None:
            Rplus = self.k*A[self.educts].prod()
            Rminus = self.k*A[self.products].prod()
        else:
            Rplus = self.k*A[self.educts].prod()*A[self.catalysts].prod()
            Rminus = self.k*A[self.products].prod()*A[self.catalysts].prod()
        
        flux = Rplus - Rminus
        
        return flux
    
    def get_potential_difference(self, A):
        
        mu_educts = np.log(A[self.educts]).sum()
        mu_products = np.log(A[self.products]).sum()
        
        delta_G = mu_products - mu_educts
        
        return delta_G
    
        
class simple_reaction_network:
    # initializers
    def __init__(self, N = 25, M = 35, p_bimolecular = 0.3, p_catalytic = 0.5, steps = int(1e6), ks = [1, 1e-1, 1e-2, 1e-3], dt = 0.001, create_reactions = True, noise = 0.0):
        
        self.N = N
        self.M = M
        
        self.A = np.zeros( (N,) )
   
        self.dt = dt
        
        self.clamped = dict()
        
        self.noise = noise
        
        self.reactions = []
        
        if create_reactions:
            # Create baseline reactions
            for i in range(M):

                # Create monomolecular reaction
                if np.random.rand() < 1.0 - p_bimolecular:
                    chemicals = np.random.choice(N,2,replace = False)
                    educt = chemicals[0]
                    product = chemicals[1]

                    self.reactions.append(reaction(N, [educt], [product], None, ks))

                # Create non-catalytic bimolecular reaction
                elif np.random.rand() < 1.0 - p_catalytic:
                    chemicals = np.random.choice(N,4,replace = False)
                    educt = chemicals[0:2]
                    product = chemicals[2:5]

                    self.reactions.append(reaction(N, educt, product, None, ks))

                # Create catalytic bimolecular reaction
                else:
                    chemicals = np.random.choice(N,5,replace = False)
                    educt = chemicals[0:2]
                    product = chemicals[2:4]
                    catalyst = chemicals[4]

                    self.reactions.append(reaction(N, educt, product, [catalyst], ks))
                                    
    def clamp_concentration(self, species, concentration):
        
        self.clamped[species] = concentration
        
    def get_potential(self, species):
        
        pot = np.log(self.A[species])
        
        return pot
    
    def get_potentials(self):
        
        pot = np.log(self.A)
        
        return pot
            
    def init_concentrations(self, A0 = None):
        
        if A0 is None:
            alpha = np.ones((self.N,))
            init_state = np.random.dirichlet(alpha)*self.N
            self.A = init_state
        else:
            self.A = A0
        
        # Enforce clamped variables, if there are any
        for key, value in self.clamped.items():
                self.A[key] = value
                
    def get_fluxes(self, A):
        
        flux = np.zeros_like(A)
        
        for r in self.reactions:
            flux += r.get_flux(A)
            
        return flux
        
    # forward method
    def run(self, steps = 100000, log_fluxes = True, show_progress_bar = True):
        
        Aout = np.zeros((steps,self.N))
        AoutG = np.zeros((steps,self.N))
        
        for key, value in self.clamped.items():
                self.A[key] = value
                
        Aout[0,:] = self.A
        AoutG[0,:] = self.A
        
        if log_fluxes:
            fluxes = np.zeros( (steps, len(self.reactions)) )
            delta_Gs = np.zeros( (steps, len(self.reactions)) )
            clamped_fluxes = np.zeros( (steps, len(self.clamped)) )
            clamped_potentials = np.zeros( (steps, len(self.clamped)) )
            potentials = np.zeros( (steps, self.N) )
            clamped_species = list(self.clamped.keys())
            
        if show_progress_bar:
            iters = tqdm(range(1,steps))
        else:
            iters = range(1,steps)

        for t in iters:
            
            Flux = self.get_fluxes(self.A)
            
            if log_fluxes:
                for i in range(len(self.reactions)):
                    fluxes[t-1,i] = self.reactions[i].get_scalar_flux(self.A)
                for i in range(len(self.reactions)):
                    delta_Gs[t-1,i] = self.reactions[i].get_potential_difference(self.A)
                for i in range(len(clamped_species)):
                    clamped_fluxes[t-1,i] = Flux[clamped_species[i]]
                    clamped_potentials[t-1,i] = self.get_potential(clamped_species[i])
                potentials[t-1,:] = self.get_potentials()
            
            self.A = self.A + self.dt*Flux
            
            AoutG[t,:] = self.A  
            
            if self.noise > 1e-12:
                self.A = self.A + self.noise*np.random.randn(self.N)
                   
            for key, value in self.clamped.items():
                self.A[key] = value
                
            Aout[t,:] = self.A     
            
            
                
            
        if log_fluxes:
            
            for i in range(len(self.reactions)):
                fluxes[-1,i] = self.reactions[i].get_scalar_flux(self.A)
            for i in range(len(self.reactions)):
                delta_Gs[-1,i] = self.reactions[i].get_potential_difference(self.A)
            for i in range(len(clamped_species)):
                    clamped_fluxes[-1,i] = Flux[clamped_species[i]]
                    clamped_potentials[-1,i] = self.get_potential(clamped_species[i])
            potentials[-1,:] = self.get_potentials()
            
            return Aout, fluxes, delta_Gs, clamped_fluxes, clamped_potentials, potentials, AoutG
        else:                           
            return Aout
    
