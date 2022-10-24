'''

This is a general purpose analysis tool for looking at the progress of assembly processes 
simulated in HOOMD. This only works when subunit binding occurs on the edges, i.e. for 
systems like the triangular subunits, and the diamond subunits. It will not directly work, 
for example, on the pentagonal subunits for dodecahedron assembly, whose attractor sites 
are on the vertices of the subunit. 

The user provides an input file that specifies the active interactions between pseudoatoms. 
If no file is provided, the code will look for a "interactions.txt" file in its directory. 
For example a line like:

A1 A3 1.0

will tell the code that pseudoatom types 'A1' and 'A3' interact with a bond length of 1 
and that distances between these atom types should be checked for distance constraints 
that determine bond formation. This reduces the total number of distance checks needed to 
determine the bond structure, dramatically speeding up the code. 

The bond network is stored in a dictionary and passed to a graph algorithm that can detect
clustering. Can compute the number of clusters of each size. 

Also supports assembly around a nanoparticle. If a line in the input text file has only
one pseudoatom type instead of two, this is assumed to be the nanoparticle ID and its radius. 
If a  nanoparticle is detected, only subunits within a cutoff of the nanoparticle are 
considered.

In this case, can output the number of attached subunits, the largest clusters size, the 
number of bonds, and the sum of cluster sizes. 

'''

import gsd.hoomd
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

import warnings
import sys
import os

from body import body
from body import neighborgrid as ng
from body import cluster as cluster

#this is a hack for now. make this more general later. WARNING: increase this size to be 
#larger than the largest structure you see in your simulations
global MAX_SIZE
MAX_SIZE = 100

'''

SimInfo Class Design

Basically a struct for simulation info that will be useful to have at various stages of the
analysis pipeline. 

I want this class to just contain information about the simulation so I do not need to 
pass things around like crazy. These include things that I was previously using global 
variables to store, such as the number of subunits and nanoparticles, box sizes, 
parameters, etc. Then I can just pass around the class and access the values as needed.

'''

class SimInfo:

    def __init__(self, snap, frames, ixn_file = "interactions.txt", cutoff_mult = 1.35, radius_mult = 1.2):

        #print an initialization message
        print("Initializing SimInfo and Neighborgrid...")

        #set the directly given variables
        self.frames = frames
        print("This simulation output contains {} frames".format(frames))
        self.cutoff_mult = cutoff_mult
        self.radius_mult = radius_mult

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
        self.num_nanos     = 0
        self.__get_num_bodies(particle_info)

        #map the interacting particle types to a hoomd integer type
        self.type_map = {}
        self.interacting_types_mapped = []
        self.__construct_map(snap)

        #get the max subunit size
        self.max_subunit_size = self.__get_max_subunit_size(snap)
        print("The largest center-to-atom distance is {}".format(self.max_subunit_size))

        #construct a neighborgrid using the sim box and info on interaction ranges
        self.ngrid = None
        self.__create_neighbor_grid()


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
                    raise ValueError("The interaction file contains a row with an unsupported"\
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
        print("The largest rest bond distance is {}".format(self.largest_bond_distance))

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
        for atom_type in self.interacting_types:

            condition = (particle_info['type'] == atom_type)
            particle_list = set(particle_info.loc[condition]['body'])
            body_set = body_set.union(particle_list)

        #set the number of bodies as the length of the body_set
        self.num_bodies = len(body_set)

        #loop over all nano types to get number of nanoparticles
        body_set = set()
        for nano_type in self.nano_types:

            ntype_mask = (particle_info['type'] == nano_type)
            nano_list  = particle_info.loc[ntype_mask]['body']

        #set the number of nanoparticles as the length of the body_set
        self.num_nanos = len(nano_list)

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

        #set the interaction range as the longest bond dist or particle size in the sim
        R = np.maximum(self.largest_bond_distance * self.cutoff_mult, 
                       self.max_subunit_size * 2 * self.radius_mult)
        print("The largest center-to-center distance for neighborgrid has been set to", R)

        #construct the neighborgrid
        self.ngrid = ng.Neighborgrid(lims, R, periodic)

        return

    def __get_max_subunit_size(self, snap):
        ''' Determine the longest midpoint to pseudoatom distance for the 
            subunits in the snap. Double this distance can be used as an interaction
            distance for the neighborgrid, which adapts to a particular systems subunits'''

        #init a list to store the 'radii' of each subunit type
        max_distance = []

        #init a listto store all center types and an example body_id
        center_types = []
        example_bodies = []

        #loop over the first num_bodies entries of particles, which contains the centers
        for i in range(self.num_bodies):

            #get the type for the centers and log an example body_id
            new_type = snap.particles.types[snap.particles.typeid[i]]
            if new_type not in center_types:
                center_types.append(new_type)
                example_bodies.append(snap.particles.body[i])

        #grab each body and get the distance to each pseudoatom
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
            for i in range(1,len(double_masked_pos)):

                particle_dist = body.distance(center_pos, double_masked_pos[i], self.box_dim)
                if (particle_dist > max_particle_dist):
                    max_particle_dist = particle_dist

            #append the max distance and body type to their arrays
            max_distance.append(max_particle_dist)
        

        # get the maximum in max_distance and return it
        max_dist_overall = np.max(np.array(max_distance))

        return max_dist_overall
        



####################################################################
################# Main Drivers #####################################
####################################################################





def analyze_structures(snap, sim, radius = None, center = None):
    #analyze clusters of subunits and their connectivity

    #get a list of bodies to analyze
    bodies = body.create_bodies(snap, sim)

    #init a dictionary to store the bonds present - init with empty lists for each body_id
    bond_dict = dict()
    for bod in bodies:
        bond_dict[bod.get_id()] = []

    #determine the bond network using the list of bodies
    body.get_bonded_bodies(bodies, sim, bond_dict)

    #debug lines
    # print(bond_dict)
    # bonds = 0
    # for key in bond_dict.keys():
    #     bonds += len(bond_dict[key])
    # print("Bonds ", bonds/2)
    # return bonds, bonds

    #determine groups of bonded structures
    G = cluster.get_groups(bond_dict)
    print(G)

    #for each group, create a cluster
    clusters = []
    for group in G:

        if len(group) > 1:
            body_list = [bodies[q] for q in group]
            print(body_list)
            clusters.append(cluster.Cluster(body_list))
            print(clusters[-1].get_body_ids())


    sys.exit()



    #count the sizes of each group
    size_counts, largest_group_size = cluster.get_group_sizes(G)
    print(size_counts)
    print(largest_group_size)
    sys.exit()

    #if nanoparticle present, compute (num_adsorbed, largestClusterSize, largestClusterBonds)
    if (radius and center.any()):

        #check if there are no clusters on the nanoparticle
        if len(G) == 0:
            return (len(bond_dict), 0, 0)

        #get largest cluster size
        G_len = [len(G[i]) for i in range(len(G))]
        largest_cluster_size = np.max(G_len)

        #get the number of bonds
        largest_cluster_id = np.argmax(G_len)
        largest_cluster    = G[largest_cluster_id]
        bonds = 0
        for particle in largest_cluster:
            bonds += len(bond_dict[particle])

        #get the number adsorbed
        num_adsorbed = np.sum(G_len)

        return (num_adsorbed, largest_cluster_size, int(bonds / 2))


    #if there is no nanoparticle, return data on the entire simulation box
    return size_counts, largest_group_size




def run_analysis(gsd_file, jump = 1, ixn_file = "interactions.txt", verbose = False, write_output = False):
    #get number of monomers and dimers at each frame in the sim

    #get the collection of snapshots and get number of frames
    snaps = gsd.hoomd.open(name=gsd_file, mode="rb")
    snap = snaps.read_frame(0)
    frames = len(snaps)

    #gather all the relevant global info into a SimInfo object
    sim = SimInfo(snap, frames, ixn_file = ixn_file)

    #create an observer to compute requested observables
    observer = cluster.Observer(gsd_file)
    observer.init_test_set()

    #init outfile to dump results
    if write_output:
        fout = open("analysis_out.dat", 'w') 

    #init an array to track live clusters
    cluster_info  = []

    #loop over each frame and perform the analysis
    f0 = 500
    old_frame = cluster.get_data_from_snap(snaps.read_frame(f0-1), sim, f0-1)
    old_frame.create_first_frame(cluster_info, f0-1, observer)
    # print(old_frame.get_clusters()[1].get_cluster_id())
    # print(old_frame.get_monomer_fraction())
    # print(cluster_info[0].get_data())
    # print(cluster_info[0].get_monomer_gain_data())
    # sys.exit()
    for frame_num in range(f0, frames, jump):

        #get the snapshot for the current frame
        snap = snaps.read_frame(frame_num)

        #check if there are nanoparticles in the simulation. If so, construct NP objects
        if (sim.nano_flag):

            nanoparticles = body.get_nanoparticles(snap, sim)

        #analyze the structures based on nanoparticle presence
        sim.nano_flag = False
        if (not sim.nano_flag):

            cluster_info, old_frame = cluster.track_clustering(snap, sim, frame_num, 
                                                                cluster_info, old_frame,
                                                                observer)

        else:
            q = []
            for nano_type in range(sim.num_nano_types):
                for nanoparticle in range(len(nano_centers[nano_type])):
                    r      = nano_radii[nano_type] * sim.radius_mult + sim.largest_bond_distance
                    # print(r)
                    center = nano_centers[nano_type][nanoparticle]
                    q_i    = analyze_structures(particle_info, sim, r, center) 
                    q.append(q_i) 

        if frame_num > 550:
            ex_data = cluster_info[0].get_data()
            print(ex_data[0]['positions'])
            sys.exit()

        #write to file
        if write_output:
            
            if (not sim.nano_flag): #ouput for no nanoparticle system
                fout.write("{} ".format(frame))
                for i in range(output_max_length):
                     f.write("{} ".format(cluster_sizes[i+1]))
                fout.write("{}".format(largest_cluster))
                fout.write("\n")

            else: #output for at least 1 nanoparticle
                fout.write("{}".format(frame))
                for nano in range(sim.num_nanos):
                    fout.write(",%s,%s,%s"%(q[nano][0], q[nano][1], q[nano][2]))
                fout.write("\n")
            

        if (verbose):
            print(frame, q)

    #close the file
    if write_output:
        fout.close()
        print("Cluster sizes written to file")

    return 


if __name__ == "__main__":

    #get the name command line arguments - will update to parser in future
    try:
        gsd_file = sys.argv[1]
        ixn_file = sys.argv[2]
        jump     = int(sys.argv[3])
    except:
        print("Usage: %s <gsd_file> <ixn_file> <frame_skip>" % sys.argv[0])
        raise

    #run analysis
    run_analysis(gsd_file, jump = jump, ixn_file = ixn_file, verbose = True, write_output = False)

