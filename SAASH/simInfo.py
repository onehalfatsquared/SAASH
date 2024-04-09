'''

SimInfo Class Design

Basically a struct for simulation info that will be useful to have at various stages of the
analysis pipeline. 

I want this class to just contain information about the simulation so I do not need to 
pass things around like crazy. These include things that I was previously using global 
variables to store, such as the number of subunits and nanoparticles, box sizes, 
parameters, etc. Then I can just pass around the class and access the values as needed.

'''


import gsd.hoomd
import numpy as np
import pandas as pd

import warnings
import sys, os

from .structure import body
from .util import neighborgrid as ng


class SimInfo:

    def __init__(self, snap, frames, ixn_file = "interactions.txt", cutoff_mult = 1.35, 
                 radius_mult = 1.6, ngrid_R = None, verbose = True):
        
        #do a verbosity check
        self.verbose = verbose
        
        #print an initialization message
        self.vprint("\nInitializing SimInfo and Neighborgrid...")

        #set the directly given variables
        self.frames = frames
        self.cutoff_mult = cutoff_mult
        self.radius_mult = radius_mult
        self.ngrid_R     = ngrid_R
        self.vprint("This simulation output contains {} frames".format(frames))

        #set the bond and nanoparticle information using ixn_file
        self.bonds = []
        self.nanos = []
        self.interacting_types = []
        self.nano_types        = []
        self.nano_flag = False
        self.largest_bond_distance = 0
        self.__parse_interactions(ixn_file)

        #get a particle info dataframe to do following computations
        particle_info = body.get_particle_info(snap)

        #determine the dimension of the system
        self.dim = 0
        self.__get_dim(particle_info)

        #set the box dimension from the snap
        box = snap.configuration.box
        if (self.dim == 2):
            self.box_dim = np.array([box[0], box[1]])
        elif (self.dim == 3):
            self.box_dim = np.array([box[0], box[1], box[2]])

        #determine the number of interacting bodies and nanoparticles from the snap
        self.num_bodies    = 0
        self.body_set      = set()
        self.body_offset   = 0
        self.num_nanos     = 0
        self.__get_num_bodies(particle_info)

        #map the interacting particle types to a hoomd integer type
        self.type_map = {}
        self.interacting_types_mapped = []
        self.__construct_map(snap)

        #get the max subunit size
        self.max_subunit_size = self.__get_max_subunit_size(snap)
        self.vprint("The largest center-to-atom distance is {}".format(self.max_subunit_size))

        #construct a neighborgrid using the sim box and info on interaction ranges
        self.ngrid = None
        self.__create_neighbor_grid()

        self.vprint("\n")

        #check if the number of particles is zero and throw error
        if (self.num_bodies == 0):
            raise ValueError("No bodies could be found based on the type naming"\
                             " in the given interaction file. Check that the interaction"\
                             " file contains correct names for the pseudoatoms. ")

        #check if nanoparticles are specified but none are found by type in ixn file
        if (self.num_nanos == 0 and self.nano_flag):
            warnings.warn("The interaction file contains a nanoparticle definition, "
                          "but no particles of this type were found in the trajectory. "\
                          "Continuing as if no nanoparticle is present. ")
            self.nano_flag = False

        return


    def __parse_interactions(self, ixn_file):
        #parse the interaction file to get a list of interacting atom types

        #check if the supplied file exists
        if (not os.path.exists(ixn_file)):

            #check if the backup files exists
            if (os.path.exists("interactions.txt")):
                ixn_file = "interactions.txt"
                warnings.warn("The specified interaction file could not be found. "\
                               "Defaulting to file 'interactions.txt'")

            else:
                raise ValueError("No interaction files could be found.")

        #append new interacting atoms and nano types to these sets
        atom_types = set()
        nano_types = set()

        #open the text file and loop over it line by line
        with open(ixn_file, 'r') as f:
            for line in f:

                #strip the newline character of the lines
                line=line.strip()

                #get the entries on the this line, space separated
                entries = line.split(' ')
                entries[-1] = float(entries[-1])

                #check the number of entries. If 3, this is a bond. If 2, nanoparticle. 
                if len(entries) == 3:

                    #construct a representation of the bond and add it to list
                    bond = body.Bond(entries[0], entries[1], entries[2])
                    self.bonds.append(bond)

                    #add the interacting particle types to the set
                    atom_types.add(entries[0])
                    atom_types.add(entries[1])

                elif len(entries) == 2:

                    #construct a representation of the nanoparticle and add it to list
                    nano = body.Nano(entries[0], entries[1])
                    self.nanos.append(nano)
                    self.nano_flag = True

                    #add the nano type to the set
                    nano_types.add(entries[0])

                else:
                    raise IndexError("The interaction file contains a row with an unsupported"\
                                     " number of entries. We require either 3 entries to define"\
                                     " a bond, or 2 entries to define a nanoparticle. ")

        #compute the number of types of bonds and nanoparticles
        self.num_bond_types = len(self.bonds)
        self.num_nano_types = len(self.nanos)

        #construct a list to store the (unique) interacting types
        self.interacting_types = list(atom_types)
        self.nano_types        = list(nano_types)

        #compute the largest bond distance in the simulation
        bond_lengths = [bond.get_cutoff() for bond in self.bonds]
        self.largest_bond_distance = np.max(bond_lengths)
        self.vprint("The largest rest bond distance is {}".format(self.largest_bond_distance))

        return

    def __get_dim(self, particle_info):
        #check if the z-coordinates are all zero, which means we have a 2d simulation

        #set a tolerance for being close to 0
        zero_tol = 1e-8

        #get the z-coordinates for all particles
        z_coords = np.array(particle_info['position_z'].values)

        #sum the absolute values and compare to 0
        S = np.sum(np.abs(z_coords))
        if (S > zero_tol):
            self.dim = 3
        else:
            self.dim = 2

        return


    def __get_num_bodies(self, particle_info):
        #get the number of subunits in the simulation by counting bodies including any 
        #of the interacting particle types. Also get the number of nanoparticles. 

        #loop over all interacting particle types to get number of bodies
        body_set = set()
        type_set = set()
        for atom_type in self.interacting_types:

            condition = (particle_info['type'] == atom_type)
            particle_list = set(particle_info.loc[condition]['body'])
            body_set = body_set.union(particle_list)
            
        #set the body id set. get offset as the first id
        self.body_set = body_set.copy()
        self.body_offset = list(self.body_set)[0]

        #set the number of bodies as the length of the body_set
        self.num_bodies = len(body_set)
        self.vprint("This simulation output contains {} subunits".format(self.num_bodies))

        #if there are multiple subunit types, get the counts for each
        types   = particle_info.loc[list(body_set)]['type']
        counter = types.value_counts()
        counter = dict(counter)

        #print the distribution of subunits if there are multiple
        self.num_subunit_types = len(counter)
        if self.num_subunit_types > 1:

            for sub_type in counter:
                self.vprint("{} subunits have type {}".format(counter[sub_type], sub_type))


        #loop over all nano types to get number of nanoparticles
        body_set = set()
        nano_list = []
        for nano_type in self.nano_types:

            ntype_mask = (particle_info['type'] == nano_type)
            nano_list  = particle_info.loc[ntype_mask]['body']

        #set the number of nanoparticles as the length of the body_set
        self.num_nanos = len(nano_list)
        self.vprint("This simulation output contains {} nanoparticles".format(self.num_nanos))

        return

    def __construct_map(self, snap):
        #construct a mapping from a string type index to integer index using snap

        #get the list of types from the snap
        type_list = snap.particles.types

        #loop over it, check for an interacting particle, map it to the index
        for i in range(len(type_list)):

            particle_type = type_list[i]

            if particle_type in self.interacting_types:
                self.type_map[particle_type] = i

            if particle_type in self.nano_types:
                self.type_map[particle_type] = i

        for p_type in self.interacting_types:

            self.interacting_types_mapped.append(self.type_map[p_type])

        return

    def __create_neighbor_grid(self):
        #construct a neighbor grid for the simulation

        #init arrays to store the bounding box limits and periodicity
        lims     = []
        periodic = []

        #loop over each dimension of the box
        for i in range(len(self.box_dim)):

            #the box goes from -L/2 to L/2
            lims.append([-self.box_dim[i] / 2.0 , self.box_dim[i] / 2.0])

            #Assume that the simulations are periodic in each dimension
            periodic.append(1)

        #check for user set neighborgrid distance
        if self.ngrid_R is not None:
            
            R = self.ngrid_R
            self.vprint("Using user specified neighborgrid distance {}".format(R))

        else:

            #set the interaction range as the longest bond dist or particle size in the sim
            R = np.maximum(self.largest_bond_distance * self.cutoff_mult, 
                           self.max_subunit_size * 2 * self.radius_mult)
            self.vprint("The largest center-to-center distance for neighborgrid has been set to {}".format(R))

        #construct the neighborgrid
        self.ngrid = ng.Neighborgrid(lims, R, periodic)

        return

    def __get_max_subunit_size(self, snap):
        ''' Determine the longest midpoint to pseudoatom distance for the 
            subunits in the snap. Double this distance can be used as an interaction
            distance for the neighborgrid, which adapts to a particular systems subunits'''

        #init a list to store the 'radii' of each subunit type
        max_distance = []

        #grab each body and get the distance to each pseudoatom
        example_bodies = list(self.body_set.copy())
        for body_id in example_bodies:

            #check that this is a rigid body
            if (body_id < 0):
                continue

            mask         = np.where(snap.particles.body == body_id)[0]
            masked_types = snap.particles.typeid[mask]
            if len(masked_types) == 0:
                continue
            center_type  = masked_types[0]

            #get positions for the masked entries
            masked_pos   = snap.particles.position[mask]
            center_pos   = masked_pos[0]
            
            #check for dimension 2
            if self.dim == 2:
                masked_pos = masked_pos[:,0:2]
                center_pos = center_pos[0:2]

            #further mask to extract only particles involved in bonds
            sub_mask = [idx for idx,x in enumerate(masked_types) 
                        if x in self.interacting_types_mapped]
            double_masked_pos = masked_pos[sub_mask]

            #init variable to keep track of max distance, iterate over particles
            max_particle_dist = 0
            for i in range(len(double_masked_pos)):

                particle_dist = body.distance(center_pos, double_masked_pos[i], self.box_dim)
                if (particle_dist > max_particle_dist):
                    max_particle_dist = particle_dist

            #append the max distance and body type to their arrays
            max_distance.append(max_particle_dist)
        

        # get the maximum in max_distance and return it
        max_dist_overall = np.max(np.array(max_distance))

        return max_dist_overall

    def multitype(self):
        #if there is one subunit type return False. If there are several, return true

        if self.num_subunit_types > 1:
            return True

        return False
        
    def vprint(self, msg):
        #only print if verbosity is true
    
        if self.verbose:
            print(msg)
    
