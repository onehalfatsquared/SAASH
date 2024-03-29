'''
This file implements a class that takes in a gsd file and state database 
(SAASH.util.state.StateRepCollection object) and fills a 
snapshot from the trajectory with the specified number of copies of a target structure 
(specified as a SAASH.util.state.State object). It then returns a gsd file with one 
frame to be used as an initial condition for a simulation. 
Note that the returned gsd file only specifies subunit centers, so the rigid bodies
in the HOOMD simulation need to be constructed from this data.

Broadly, the approach is to randomly select, from the free-monomer population, enough
subunits to construct the target, discretize the domain into grid cells with sizes to
contain the target, and place N copies of the target in these cells such that there is 
no overlap of subunits. 

The use case of this operation is to create initial conditions for targetted sampling
of self-assembly dynamics at particular monomer fractions in such a way as to not bias the equilibrium distribution of intermediates. Particularly in the case of large 
nucleation barriers, the equilibrium distribution will be peaked at a mix of monomers
and completely assembled structures, so we just move probability between these two 
without modifying the distribution of intermediates. 

'''

import gsd.hoomd
import numpy as np
import pickle

import sys, os
import pathlib, glob

from ..structure.frame import get_data_from_snap
from ..util.state import State
from ..simInfo import SimInfo


class InitialConfigurationGenerator:
    '''
    Takes in locations of a RepCollection, an example gsd file, and a target state
    contained in the RepCollection. The create_new_config(N) method can be used 
    to add N copies of the target structure to the simulation box, using up a random
    selection of free-monomers to construct it. Saves the new configuration as a 
    one frame gsd file.
    '''

    def __init__(self, db_loc, gsd_file, target_state, 
                 ixn_file = "interactions.txt", frame = -1):

        #save the target state
        self.target_state = target_state

        # Load the state database and get the Rep of the target
        self.state_dict = self._make_state_db(db_loc)
        self.target_state_rep = self.state_dict[target_state]

        # Load the final snapshot as reference
        with gsd.hoomd.open(name=gsd_file, mode="rb") as gsd_file:
            self.ref_snap = gsd_file[frame]

        # Create a SimInfo object and extract needed details from the system
        self.siminfo  = SimInfo(self.ref_snap, 1, ixn_file=ixn_file, verbose=False)
        self.box_size = self.ref_snap.configuration.box[0:3]
        self.num_subs = self.siminfo.num_bodies
        self.monomers = list(get_data_from_snap(self.ref_snap, self.siminfo, -1).get_monomer_ids())

        #get the monomer fraction, and the fraction of monomers a target takes up
        self.mon_frac = float(len(self.monomers)) / float(self.num_subs)
        self.target_frac = float(self.target_state.get_size()) / float(self.num_subs)

        #get the max number of target structures that can be placed
        self.max_targets = self.get_max_additions()

        #get a map from number of target structures to new monomer fraction
        self._make_frac_map()

        #set parameters used in the processing
        self.__widening_factor = 1.1 #multiply bounding box of target by this 

        #define a bounding box for the target state using its positions
        self._bound_target_config()

        #setup the grid for counting subunits within the domain
        self._setup_grid()

        return
    
    def print_possible_fractions(self):
        #print what the resultant monomer fraction would be as a fn of num targets added

        sorted_num = sorted(list(self.num_to_frac.keys()))
        for num in sorted_num:
            current_frac = self.num_to_frac[num]
            print("Adding {} targets gives monomer fraction {}".format(num, current_frac))

        return

    def get_matching_num(self, frac):
        #return the num of targets to add to get monomer fraction closest to the input

        closest_key = None
        closest_distance = 1000

        for key, value in self.num_to_frac.items():
            distance = abs(value - frac)
            if distance < closest_distance:
                closest_distance = distance
                closest_key = key

        return closest_key

    def get_max_additions(self):
        #determine the max number of targets that can be added

        return int(self.mon_frac / self.target_frac)
    
    def _make_frac_map(self):
        #make a dict that stores what the resulting mon frac is as fn of num targets

        self.num_to_frac = dict()
        for i in range(self.max_targets+1):

            self.num_to_frac[i] = round(self.mon_frac - i * self.target_frac,2)

        return

    def _make_state_db(self, db_loc):
        #extract dict mapping of States to Reps

        # Open the pickled database
        with open(db_loc, 'rb') as f:
            ref_collect = pickle.load(f)

        # Extract the dictionary of states from the database
        return ref_collect.get_dict()

    def _select_monomers(self, num_to_add):
        #determine how many monomers to make into targets, randomly sample and assign them
    
        #figure out how many to sample
        monomers_per_state = self.target_state.get_size()
        monomers_to_extract= monomers_per_state * num_to_add

        #do the sampling, assign each monomer to a target state copy
        sampled_monomers   = np.random.choice(self.monomers, monomers_to_extract, False)
        new_target_mon_ids = np.split(sampled_monomers, num_to_add)

        #remove sampled monomers from the list of remaining monomers
        remaining = [idx for idx in self.monomers if idx not in sampled_monomers]

        return new_target_mon_ids, remaining

    def _bound_target_config(self):
        #determine a box length big enough to contain the target state

        #determine the bounding box side length
        positions = self.target_state_rep.get_positions()
        m = np.min(positions, 0)
        M = np.max(positions, 0)
        self._bounding_box_L = np.max(M-m) + self.siminfo.max_subunit_size
        self._bounding_box_L*= self.__widening_factor

        return 
    
    def _setup_grid(self):
        #determine augmented bounding box and grid properties

        #first determine the uniform gridding that is closest to L but is still bounding
        nx, ny, nz = (self.box_size / self._bounding_box_L).astype(int)
        dx, dy, dz = self.box_size / [nx,ny,nz]

        #store num and length of grid cells. store half box sizes for shifting
        self._grid_numbers = np.array([nx,ny,nz])
        self._grid_lengths = np.array([dx, dy, dz])
        self.box_half      = self.box_size / 2

        return
    
    def _position_to_grid(self, pos):
        #convert a position to the grid cell containing it

        return [int((pos[i] + self.box_half[i])/self._grid_lengths[i]) for i in range(3)]

    def _grid_to_center(self, grid_cells):
        #convert a grid cell to the center position of that cell

        return ((grid_cells+0.5) * self._grid_lengths - self.box_half)
    
    def _assign_subs_to_box(self):
        #count number of subunits/monomers in each subgrid defined by bounding box of target

        #init storage for counting subunits and removed monomers from each box
        box_sub_counts = np.zeros(self._grid_numbers, dtype=int)
        rem_mon_counts = np.zeros(self._grid_numbers, dtype=int)
        large_counts   = np.zeros(self._grid_numbers, dtype=int)

        #loop over particles and assign them to a box
        for p_idx in range(self.num_subs):

            #determine grid cell containing this particle position
            pos = self.snap.particles.position[p_idx]
            i,j,k = self._position_to_grid(pos)

            #increment box counts
            box_sub_counts[i,j,k] += 1

            #separately track how many monomers are removed from each box
            if p_idx in self.monomers and p_idx not in self._remaining:
                rem_mon_counts[i,j,k] += 1

            #separetely track where large structures are
            if p_idx not in self.monomers:
                large_counts[i,j,k] += 1

        return box_sub_counts, rem_mon_counts, large_counts
    
    def _check_adjacency(self, indices, sub_counts, poss_indices):
        #check that cells adjacent to given indices do not have large intermediates

        #get dimensions of the array and extract the indices
        ni, nj, nk = sub_counts.shape
        i, j, k    = indices

        #get all adjacent values, making sure to handle pbcs
        adjacent_values = [
            sub_counts[(i - 1) % ni, j, k],
            sub_counts[(i + 1) % ni, j, k],
            sub_counts[i, (j - 1) % nj, k],
            sub_counts[i, (j + 1) % nj, k],
            sub_counts[i, j, (k - 1) % nk],
            sub_counts[i, j, (k + 1) % nk],
        ]

        #check that they are all less than 1
        if np.max(adjacent_values) > 0:
            return False
        
        #check that this cell is not adjacent to any other cell we have
        for valid_index in poss_indices:
            S = (np.array(indices) - np.array(valid_index)) % sub_counts.shape
            S[S==3] = 1
            if S.sum() == 1:
                return False
        
        return True

    def _choose_target_location(self, num_to_add):
        #determine a location to place the new target structures

        #first, discretize domain into boxes, count how many subs and mons are in each
        sub_counts, rem_counts, large_counts = self._assign_subs_to_box()

        #compute the number of effective subs in each box by subtracting
        eff_subs = sub_counts - rem_counts

        #check for boxes where at most one monomer remains
        poss_boxes = np.where(eff_subs <= 1)
        poss_indices = []
        for j in range(len(poss_boxes[0])):

            #get indices of test box
            idx_test = [poss_boxes[0][j], poss_boxes[1][j], poss_boxes[2][j]]

            #check if it is not adjacent to cell with larger intermediates
            remote = self._check_adjacency(idx_test, large_counts, poss_indices)

            #if the cell is remote, add this possibility
            if remote:
                poss_indices.append(idx_test)

        #uniformly sample the potential bounding boxes
        sampled_indices   = np.random.choice(range(len(poss_indices)), num_to_add, False)
        sampled_grid_cells= np.array([poss_indices[i] for i in sampled_indices])

        #convert these grid cells into the center point of the grid cell
        center_points = [self._grid_to_center(sampled_grid_cells[i]) for i in range(num_to_add)]

        return center_points
    
    def _optimize_step_size(self, particle_id, direction):
        #determine how much to move the specified particle to maximize its minimum distance to other particles

        #set a step size and compute how many trials should be done
        step_inc = 0.10
        max_dist = np.min(self.box_size)
        num_iters = int(max_dist/step_inc)

        #set defaults for best objective value and position
        obj = -1
        best_pos = None

        #set shortcut for unmodified positions list and remove current particle
        fixed_locs = self.snap.particles.position[0:self.num_subs]
        fixed_locs = np.delete(fixed_locs, particle_id, axis=0)

        #perform the line search
        for i in range(num_iters):

            #get the updated particle pos
            x = self.snap.particles.position[particle_id] + i * step_inc * direction

            #check for moving across pbc 
            x = ((x+self.box_half) % self.box_size) - self.box_half

            #eval distance to all particles, get minimum
            sq_dists = distance2(x, fixed_locs, self.box_size)
            m = np.min(sq_dists)

            #check if better than previous
            if m > obj:
                obj = m
                best_pos = x

        return best_pos
    
    def _reduce_overlap(self, center_loc, monomer_indices):
        #determine all particles too close to the center that are not part of target
        #if they are monomers, move them

        self.overlap_cut = 0.85 * self._bounding_box_L * self._bounding_box_L

        #get squared distances 
        sq_dists = distance2(center_loc, self.snap.particles.position[0:self.num_subs],
                             self.box_size)
        
        #find distances less than cutoff, remove monomers in the target state
        within_cut = np.where(sq_dists < self.overlap_cut)[0]
        within_cut = list(set(within_cut).difference(set(monomer_indices)))

        #check if each of these is a monomer, if so, move it
        for p_id in within_cut:
            if p_id in self._remaining:

                #get the direction from target center to particle
                normal_dir = self.snap.particles.position[p_id] - center_loc
                normal_dir /= np.linalg.norm(normal_dir)

                #pick a step size to move it to maximize the min distance to the nearest particle
                best_loc = self._optimize_step_size(p_id, normal_dir)
                self.snap.particles.position[p_id] = best_loc
        
        return

    def _place_target(self, target_centers):
        #take selected monomers and form a target structure. Place at provided centers

        #get the positions and orientations of subunits in the target
        target_positions = self.target_state_rep.get_positions()
        target_orientations = self.target_state_rep.get_orientations()

        #loop over each structure to add, modify the snap
        for target_num in range(len(target_centers)):

            #get the location and monomers involved in this construction
            center_location = target_centers[target_num]
            monomer_indices = self._new_target_mons[target_num]

            #modify each index in the snap, give it the corresponding data
            count = 0
            for m_idx in monomer_indices:
                
                #get the new position and set it. set new orientation
                new_pos = target_positions[count] + center_location
                new_ori = target_orientations[count]
                self.snap.particles.position[m_idx] = new_pos
                self.snap.particles.orientation[m_idx] = new_ori

                count += 1

            #move any monomers that may overlap with the new target structure
            self._reduce_overlap(center_location, monomer_indices)

        return 
    
    def _make_new_snap(self, old_snap):
        #makes a new HOOMD Frame and fills it with info from the provided snap

        #create new frame
        nsnap = gsd.hoomd.Frame()

        ns = self.num_subs #shortcut for number of subunits
        #fill it with required info. only include info for centers
        nsnap.particles.N              = self.num_subs
        nsnap.configuration.box        = old_snap.configuration.box
        nsnap.particles.types          = old_snap.particles.types
        nsnap.particles.position       = old_snap.particles.position[0:ns].copy()
        nsnap.particles.orientation    = old_snap.particles.orientation[0:ns].copy()
        nsnap.particles.typeid         = old_snap.particles.typeid[0:ns]
        nsnap.particles.moment_inertia = old_snap.particles.moment_inertia[0:ns]

        return nsnap
    
    def _print_result(self, num_to_add):
        #print a message upon succesful generation

        mon_frac = self.num_to_frac[num_to_add]
        print("A new configuration was generated with {} additional copies of the structure...".format(num_to_add))
        print(self.target_state)
        print("The new monomer fraction is {}.".format(mon_frac))

        return


    def create_new_config(self, num_to_add, verbose=False):
        #take the initial frame, replace random monomers with specified number
        #of target structures

        #do input checking
        num_to_add = int(num_to_add)
        if num_to_add < 0:
            err_msg = "Please specify a non-negative number of targets to place. "
            raise ValueError(err_msg)
        elif num_to_add > self.max_targets:
            err_msg = "This snapshot can only support adding up to {} targets. You specified {}.".format(self.max_targets, num_to_add)
            raise ValueError(err_msg)

        #create a new snap to modify to make the new config
        self.snap = self._make_new_snap(self.ref_snap)
        
        #place targets if a positive integer is given
        if num_to_add > 0:
        
            #choose monomer ids that will become the target, and which remain
            self._new_target_mons, self._remaining = self._select_monomers(num_to_add)

            #determine a location(s) to place the target
            target_centers = self._choose_target_location(num_to_add)

            #modify the snap to place the new structures
            self._place_target(target_centers)

        #make a new snap containing the minimum info needed to init a simulation
        new_snap = self._make_new_snap(self.snap)

        #optional final message
        if verbose:
            self._print_result(num_to_add)

        return new_snap

class BatchConfigurationGenerator:
    '''
    The class takes in a database of states and a target state, as well as a folder
    of simulations to use as a base. It loops over each trajectory in the folder,
    extracts the specified frame from it, and uses it as a base to generate 
    configurations at various monomer fraction values. 

    It then saves the configurations to appropriate folders separated by monomer 
    fraction and labeled descriptively. 
    '''

    def __init__(self, db_loc, base_folder, target_state, 
                 ixn_file = "interactions.txt", frame = -1,
                 traj_file = "*.gsd", exclusions = []):

        #store all inputs so they can be used to construct objects within methods
        self.db_loc = db_loc
        self.base_folder = base_folder
        self.target_state = target_state
        self.ixn_file = ixn_file
        self.frame = frame
        self.traj_file = traj_file

        self.verbose = False

        #set file name fragments to exclude from the search
        self._set_exclusions(exclusions)

        #check for and init paths for files generated by this class
        self._prepare_paths(base_folder)

        #set variables to store number of attempted and failed creation attempts
        self._failures = 0
        self._attempts = 0

        return
    
    def _set_exclusions(self, exclusions):
        #set base exclusions and augment with any user provided exclusions

        #store the base exclusion file name fragments
        self.exclusions = ["equilibrated_start", "lattice"]

        #append any user specified exclusions
        for e in exclusions:
            self.exclusions.append(e)

        return
    
    def _prepare_paths(self, folder):
        #init directory structure if tirst time through. set save locations
        #save locations of all files to be processed

        #check that this folder is acceptable to run generate on
        self._verify_base(folder)

        #check for the initial config directory, make if not found
        self.config_folder = folder + "initial_configs/"
        if not os.path.exists(self.config_folder):
            os.makedirs(self.config_folder)

        #traverse into this directory to check for subdirs. make dict refs to them
        self.save_locations = dict()
        for i in range(10):
            frac_folder = self.config_folder+"frac{}/".format(i)
            self.save_locations[i] = frac_folder
            if not os.path.exists(frac_folder):
                os.makedirs(frac_folder)

        #get paths to all gsd files to be used as a base to generate ICs
        gsd_files = sorted(glob.glob(folder+"*/"+self.traj_file))

        #filter files with excluded names and save to a class variable
        self.base_files = [f for f in gsd_files if not any(phrase in f for phrase in self.exclusions)]

        return
    
    def _verify_base(self, folder):
        #check that the supplied folder has no gsd files, but subdirs do

        #get all gsds in the folder, and in sub folders
        gsd_list     = list(pathlib.Path(folder).glob(self.traj_file))
        sub_gsd_list = list(pathlib.Path(folder).glob("*/"+self.traj_file))

        #check that this folder has none, and that subdirs have at least 1
        if len(gsd_list) > 0:
            err_msg = "The provided path to the trajectory folder ({}".format(folder)
            err_msg+= ") contains .gsd files. Ensure this was the intended folder, "
            err_msg+= "and remove any gsd files if so."
            raise RuntimeError(err_msg)
        
        if len(sub_gsd_list) == 0:
            err_msg = "No gsd files were found in subdirectories of {}\n".format(folder)
            raise RuntimeError(err_msg)
        
        return
    
    def _get_save_loc(self, frac):
        #return the folder that hosts configs of the given frac

        folder_level = int(frac * 10)
        return self.save_locations[folder_level]
    
    def _print_run_results(self):
        #print some basic information about the run

        print("\nIC Generation Results:")
        #print info on success rate of config attempts
        success_rate = (self._attempts-self._failures) / self._attempts
        print("Configurations attempted: {}\nConfigurations failed: {}".format(self._attempts, self._failures))
        print("Success Rate: {}".format(success_rate))

        #print current coarse distribution of monomer fractions ics
        self.count_configs()

        return
    
    def count_configs(self, refined = False):
        #count the number of existing configurations saved so far

        #init a dict to store the distribution 
        if refined:
            ic_distribution = {i:self._count_refined(float(i)/100.) for i in range(100)}
        else:
            ic_distribution = {i:sum(1 for _ in pathlib.Path(self._get_save_loc(float(i)/100.)).glob('*')) for i in range(0,100,10)}

        #print the distribution out
        sorted_keys = sorted(list(ic_distribution.keys()))
        for key in sorted_keys:
            print("Fraction: {}, Num Files: {}".format(key, ic_distribution[key]))

        return ic_distribution

    def _count_refined(self, frac):
        #count the number of configurations labeled with a given fraction

        #get the folder containing this fraction
        save_loc = self._get_save_loc(frac)

        #convert frac to 2 dec places to search for files with this tag
        frac2 = int(frac * 100)
        search_prefix = "ic_{}*".format(frac2)

        #search for this prefix in this folder and return the number of files
        count_list    = list(pathlib.Path(save_loc).glob(search_prefix))
        return len(count_list)

    def _save_config(self, config, new_frac):
        #write the supplied configuration to a file

        #do nothing if creation failed
        if config is None:
            return
        
        #get the save location for this fraction, the refined fraction, and the numeric index of it
        save_loc = self._get_save_loc(new_frac)
        frac2 = int(new_frac * 100)
        label = self._count_refined(new_frac)
        save_loc += "ic_{}_{}.gsd".format(frac2, label)

        #print save location message
        if self.verbose:
            print("Configuration saved at...\n{}".format(save_loc))

        #actually save the file
        with gsd.hoomd.open(name=save_loc, mode='wb') as f:
            f.append(config)

        return

    def _create_new_config(self, ic_gen, num_to_add):
        #put the configuration creation in try-except clause, return None on fail

        #increment counter of numbe rof attempts
        self._attempts += 1

        #try to make new config. if fail, increment failures and return None
        try:
            
            return ic_gen.create_new_config(num_to_add, self.verbose)
        
        except:

            self._failures += 1
            return None
        
        return
    
    def _make_ic(self, ic_gen, num_to_add):
        #create the new initial condition and save it

        result_frac= ic_gen.num_to_frac[num_to_add]
        new_config = self._create_new_config(ic_gen, num_to_add)
        if new_config is not None:
            self._save_config(new_config, result_frac)

        return

    def _gen_configs_from_traj(self, traj_file, target_mon_frac):
        #do the generation from the given file

        #create the ICgen object
        ic_gen = ICGen(self.db_loc, traj_file, self.target_state, self.ixn_file, 
                       self.frame)
        
        #if target mon frac is specified, get the number to add to get closest
        if target_mon_frac is not None:
            num_to_add = ic_gen.get_matching_num(target_mon_frac)
            self._make_ic(ic_gen, num_to_add)

        #otherwise, loop over all number to add
        else:   
            max_additions = ic_gen.get_max_additions()
            for num_to_add in range(max_additions+1):
                self._make_ic(ic_gen, num_to_add)

        return

    def generate_configs(self, target_mon_frac = None, verbose=False):
        #loop over all the files and generate the asked for configurations

        #change the verbose flag for this computation 
        old_verbose = self.verbose
        self.verbose = verbose

        #loop over all pre-determined files as a base to add from
        for gsd_file in self.base_files:
            self._gen_configs_from_traj(gsd_file, target_mon_frac)

        #print details of the run if requested
        if self.verbose:
            self._print_run_results()

        #reset verbose flag for future calls
        self.verbose = old_verbose
        return
    
def distance2(x0, x1, dimensions):
    #get the squared distance between the points x0 and x1
    #assumes periodic BC with box dimensions given in dimensions

    #get distance between particles in each dimension
    delta = np.abs(x0 - x1)

    #if distance is further than half the box, use the closer image
    delta = np.where(delta > 0.5 * dimensions, delta - dimensions, delta)

    #compute and return the distance between the correct set of images
    return (delta ** 2).sum(axis=-1)
