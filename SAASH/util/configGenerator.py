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

        # Load the final snapshot
        with gsd.hoomd.open(name=gsd_file, mode="rb") as gsd_file:
            self.snap = gsd_file[frame]

        # Create a SimInfo object and extract needed details from the system
        self.siminfo  = SimInfo(self.snap, 1, ixn_file=ixn_file, verbose=False)
        self.box_size = self.snap.configuration.box[0:3]
        self.num_subs = self.siminfo.num_bodies
        self.monomers = list(get_data_from_snap(self.snap, self.siminfo, -1).get_monomer_ids())

        #get the monomer fraction, and the fraction of monomers a target takes up
        self.mon_frac = float(len(self.monomers)) / float(self.num_subs)
        self.target_frac = float(self.target_state.get_size()) / float(self.num_subs)

        #get the max number of target structures that can be placed
        self.max_targets = self.get_max_additions()

        #get a map from number of target structures to new monomer fraction
        self._make_frac_map()

        #set parameters used in the processing
        self.__widening_factor = 1.1 #multiply bounding box of target by this 

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
    
    def _make_new_snap(self):
        #makes a new HOOMD Frame and fills it with info from the modified snap

        #create new frame
        new_snap = gsd.hoomd.Frame()

        #fill it with required info. only include info for centers
        new_snap.particles.N = self.num_subs
        new_snap.configuration.box = self.snap.configuration.box
        new_snap.particles.types = self.snap.particles.types
        new_snap.particles.position = self.snap.particles.position[0:self.num_subs]
        new_snap.particles.orientation = self.snap.particles.orientation[0:self.num_subs]
        new_snap.particles.typeid = self.snap.particles.typeid[0:self.num_subs]
        new_snap.particles.moment_inertia = self.snap.particles.moment_inertia[0:self.num_subs]

        return new_snap
    
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
        
        #place targets if a positive integer in given
        if num_to_add > 0:
        
            #choose monomer ids that will become the target, and which remain
            self._new_target_mons, self._remaining = self._select_monomers(num_to_add)

            #define a bounding box for the target state using its positions
            self._bound_target_config()

            #setup the grid for counting subunits within the domain
            self._setup_grid()

            #determine a location(s) to place the target
            target_centers = self._choose_target_location(num_to_add)

            #modify the snap to place the new structures
            self._place_target(target_centers)

        #make a new snap containing the minimum info needed to init a simulation
        new_snap = self._make_new_snap()

        #optional final message
        if verbose:
            self._print_result(num_to_add)

        return new_snap
    
def distance2(x0, x1, dimensions):
    #get the squared distance between the points x0 and x1
    #assumes periodic BC with box dimensions given in dimensions

    #get distance between particles in each dimension
    delta = np.abs(x0 - x1)

    #if distance is further than half the box, use the closer image
    delta = np.where(delta > 0.5 * dimensions, delta - dimensions, delta)

    #compute and return the distance between the correct set of images
    return (delta ** 2).sum(axis=-1)
