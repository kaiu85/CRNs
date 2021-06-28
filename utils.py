import numpy as np
import glob
from tqdm import tqdm

def save_results(results, prefix, suffix):
    
    ts = results[0].numpy()
    As = results[1].numpy()
    updates = results[2].numpy()
    Rs = results[3].numpy()
    Ps = results[4].numpy()
    reactions = results[5].numpy()

    np.save(prefix + 'ts' + suffix + '.npy',ts)
    np.save(prefix + 'As' + suffix + '.npy',As)
    np.save(prefix + 'updates' + suffix + '.npy',updates)
    np.save(prefix + 'Rs' + suffix + '.npy',Rs)
    np.save(prefix + 'Ps' + suffix + '.npy',Ps)
    np.save(prefix + 'reactions' + suffix + '.npy',reactions)
    
    return

def save_trajectories(results, prefix, suffix):
    
    ts = results[0].numpy()
    As = results[1].numpy()   

    np.save(prefix + 'ts' + suffix + '.npy',ts)
    np.save(prefix + 'As' + suffix + '.npy',As)
        
    return

def save_trajectories_and_reactions(results, prefix, suffix):
    
    ts = results[0].numpy()
    As = results[1].numpy()   
    reactions = results[5].numpy()
    
    np.save(prefix + 'ts' + suffix + '.npy',ts)
    np.save(prefix + 'As' + suffix + '.npy',As)
    np.save(prefix + 'reactions' + suffix + '.npy',reactions)
        
    return

def load_list(prefix, suffixes, variable):
    
    if len(suffixes) > 0:
        results = [ np.load(prefix + variable + suffixes[0] + '.npy') ]

        for i in tqdm(range(1,len(suffixes))):
            results.append( np.load(prefix + variable + suffixes[i] + '.npy') )
            
        result = np.concatenate( results, axis = -1 )
            
        return result
    
def load_chunked(prefix, suffix, n_seq = 0, variables = ['ts', 'As', 'Rs', 'Ps', 'reactions']):
    
    # find out final number of sequences from filenames
    if n_seq == 0:
        glob_str = prefix + 'ts' + suffix + '_0_of_*.npy'    
        first_ts_file = glob.glob(glob_str)
        number_string = first_ts_file[0].split('.')[-2].split('_')[-1]    
        n_seq = int(number_string)
        
    # find out number of completed sequences
    n_done = len( glob.glob( prefix + 'ts' + suffix + '_*_of_*.npy' ) )
    
    if n_done < n_seq:
        print( 'Loading %d of %d parts...' % (n_done, n_seq) )
    
    suffixes = []
    for i in range(n_done):
        suffixes.append(suffix + '_' + str(i) + '_of_' + str(n_seq))
    
    results = []
    
    for var in tqdm(variables):
        results.append( load_list(prefix, suffixes, var) )
    
    return results
