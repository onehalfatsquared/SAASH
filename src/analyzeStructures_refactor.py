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

        #set the directly given variables
        self.frames = frames
        self.cutoff_mult = cutoff_mult
        self.radius_mult = radius_mult

        #set the bond and nanoparticle information using ixn_file
        self.bonds = []
        self.nanos = []
        self.nano_flag = False
        self.largest_bond_distance = 0
        self.__parse_interactions(ixn_file)

        #set the box dimension from the snap
        box = snap.configuration.box
        self.box_dim = np.array([box[0], box[1], box[2]])

        #determine the number of interacting bodies and nanoparticles from the snap
        self.num_bodies    = 0
        self.num_nanos     = 0
        self.__get_num_bodies(snap)

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

                elif len(entries) == 2:

                    #construct a representation of the nanoparticle and add it to list
                    nano = body.Nano(entries[0], entries[1])
                    self.nanos.append(nano)
                    self.nano_flag = True

                else:
                    raise ValueError("The interaction file contains a row with an unsupported"\
                                     " number of entries. We require either 3 entries to define"\
                                     " a bond, or 2 entries to define a nanoparticle. ")

        #compute the number of each type of bond and nanoparticle
        self.num_bond_types = len(self.bonds)
        self.num_nano_types = len(self.nanos)

        #compute the largest bond distance in the simulation
        bond_lengths = [bond.get_cutoff() for bond in self.bonds]
        self.largest_bond_distance = np.max(bond_lengths)

        return


    def __get_num_bodies(self, snap):
        #get the number of subunits in the simulation by counting bodies including any 
        #of the interacting particle types. Also get the number of nanoparticles. 

        #extract particle info from the given snap
        particle_info = get_particles(snap)

        #determine all relevant atom types in the simulation by looping over bond types
        atom_types = set()
        for bond in self.bonds:

            #get the pair of types involved in a bond - tuple
            p_types = bond.get_types()

            #add the types to the set
            atom_types.add(p_types[0])
            atom_types.add(p_types[1])

        #loop over all interacting particle types
        body_set = set()
        for atom_type in atom_types:

            condition = (particle_info['type'] == atom_type)
            particle_list = set(particle_info.loc[condition]['body'])
            body_set = body_set.union(particle_list)

        #set the number of bodies as the length of the body_set
        self.num_bodies = len(body_set)

        #loop over nanoparticle entries to add types to array
        nano_types = []
        for nano in self.nanos:
            nano_types.append(nano.get_type())

        #loop over all nano types
        body_set = set()
        for nano_type in nano_types:

            ntype_mask = (particle_info['type'] == nano_type)
            nano_list  = particle_info.loc[ntype_mask]['body']

        #set the number of particles as the length of the body_set
        self.num_nanos = len(nano_list)

        return


####################################################################
################# Utility and Data Extraction ######################
####################################################################
            

def distance(x0, x1, dimensions):
    #get the distance between the points x0 and x1
    #assumes periodic BC with box dimensions given in dimensions

    #get distance between particles in each dimension
    delta = np.abs(x0 - x1)

    #if distance is further than half the box, use the closer image
    delta = np.where(delta > 0.5 * dimensions, delta - dimensions, delta)

    #compute and return the distance between the correct set of images
    return np.sqrt((delta ** 2).sum(axis=-1))


def get_particles(snap):
    #return the needed info to track assembly from a trajectory snap as a DataFrame

    #gather the relevant data for each particle into a dictionary
    #Note: positions need to be seperated in each coordinate
    particle_info = {
        'type': [snap.particles.types[typeid] 
                 for typeid in snap.particles.typeid],
        'body': snap.particles.body,
        'position_x': snap.particles.position[:, 0],
        'position_y': snap.particles.position[:, 1],
        'position_z': snap.particles.position[:, 2],
    }

    #return a dataframe with the relevant info for each particle
    return pd.DataFrame(particle_info)



####################################################################
################# Determine Bond Network ###########################
####################################################################


def get_bonded_subunits(p1_coords, p1_bods, p2_coords, p2_bods, bond_dict, box_dim, cutoff):
    #get the subunit indices that have bonds between the two particle sets

    #todo - skip this if no nanoparticle

    #add all the relevant bodies to the bond_dict as keys
    for body in p1_bods:
        if body not in bond_dict:
            bond_dict[body] = []

    for body in p2_bods:
        if body not in bond_dict:
            bond_dict[body] = []
   

    #loop over the coordinates for particle1, comparing to all particle2's
    for i in range(len(p1_coords)):

        #get the i-th particle1, and i+1 to end particle2's
        base_coord = p1_coords[i]
        compare_coords = p2_coords

        #compute pairwise distances and compare to cutoff to get interacting pairs
        dists = distance(base_coord, compare_coords, box_dim)
        interacting = np.where(dists < cutoff)[0]

        #add all bonded pairs to the dictionary
        for sub_id in interacting:

            #do not allow self interactions
            if p2_bods[sub_id] != p1_bods[i]:

                #do not allow repeats in the list of bonded particles
                if p2_bods[sub_id] not in bond_dict[p1_bods[i]]:
                    bond_dict[p1_bods[i]].append(p2_bods[sub_id])
                if p1_bods[i] not in bond_dict[p2_bods[sub_id]]:
                    bond_dict[p2_bods[sub_id]].append(p1_bods[i])


    return
        

def get_type_coords(particle_info, particle_type, box_dim, radius = None, center = None):
    #get the list of coordinates for all particles of the given type

    #get the list of all particle info for that type
    particle_list = particle_info.loc[(particle_info['type'] == particle_type)]

    #extract the coordinates from this list of data
    p_coords = np.array([np.array(particle_list['position_x'].values), 
                         np.array(particle_list['position_y'].values), 
                         np.array(particle_list['position_z'].values)]).T

    bodies = np.array(particle_list['body'])

    #if a cutoff sphere is given, exclude particles outside this sphere
    if (radius and center.any()):
        distance_mask = np.where(distance(p_coords, center, box_dim) < radius)
        p_coords = p_coords[distance_mask]
        bodies   = bodies[distance_mask]

    #return the coordinates and body id
    return p_coords, bodies


def get_p_q_bonds(particle_info, type_p, type_q, bond_dict, box_dim, cutoff,
                  radius = None, center = None):
    #determine all pairs of particles of type p and q that are bonded

    #get the coordinates for these particle types
    p_coords, p_bods = get_type_coords(particle_info, type_p, box_dim, radius, center)
    q_coords, q_bods = get_type_coords(particle_info, type_q, box_dim, radius, center)

    #use coordinates to determine which bodies have an interacting pair
    get_bonded_subunits(p_coords, p_bods, q_coords, q_bods, bond_dict, box_dim, cutoff)

    return

####################################################################
################# Bond Network -> Graph & Clustering ###############
####################################################################

def get_groups(bond_dict):
    #construct arrays containing groups of all bonded particles

    #get 'state' info
    states = list(bond_dict.keys())
    total_states = len(states)

    #if no bonds, return 0
    if total_states == 0:
        return [], bond_dict

    #convert the indices in the dict to 1:total_states
    new_bond_dict = dict()
    for i in range(total_states):

        #get the list from the original bond_dict
        entries = bond_dict[states[i]]

        #convert the entries to new indices
        new_entries = []
        for entry in entries:
            new_entries.append(np.where(states == entry)[0][0])

        #add new entries to new dict
        new_bond_dict[i] = new_entries

    #replace old dict
    bond_dict = new_bond_dict

    #init an array to store mappings to groups
    to_group = -1 * np.ones(total_states, dtype=int)
    to_group[0] = 0

    #init a list to store the groups
    G = [[0]]

    #init a queue and dump for searching bonds
    queue    = [0]
    searched = []

    #loop over the queue until it is empty or all bonds have been assigned
    while len(queue) > 0:

        #pop the front of the queue and get its group number
        particle = queue.pop(0)
        group_num = to_group[particle]

        #get the adjacency from bond_dict for this particle
        adj = bond_dict[particle]

        #for each adjacent particle, add it to the group, set the map, add to queue
        for member in adj:

            #if the member is not in the group yet, add it and set the mapping
            if member not in G[group_num]:
                G[group_num].append(member)
                to_group[member] = group_num

            #if the member has not been searched and is not in queue, add
            if member not in searched and member not in queue:
                queue.append(member)

        #append the particle to the searched array
        searched.append(particle)

        #check if the queue is empty but there are undetermined states
        if len(queue) == 0 and len(searched) != total_states:

            #determine the first unassigned state
            unassigned = np.where(to_group == -1)[0][0]

            #create a new group for it, assign it, append to queue
            G.append([unassigned])
            to_group[unassigned] = len(G)-1
            queue.append(unassigned)

    #return the list of grouped particles
    return G, bond_dict


def get_group_sizes(G):

    #init an array of zeros up to MAX_SIZE
    size_counts = np.zeros(MAX_SIZE, dtype=int)

    #loop over groups and increment the corresponding size index
    for group in G:

        L = len(group)
        size_counts[L] += 1

    #determine the largest group
    non_zero_counts = np.where(size_counts > 0)[0]
    if len(non_zero_counts) > 0:
        largest_group_size = np.max(non_zero_counts)
    else:
        largest_group_size = 0

    #return size counts
    return size_counts, largest_group_size


####################################################################
################# Main Drivers #####################################
####################################################################


def analyze_structures(particle_info, sim, radius = None, center = None):
    #analyze clusters of subunits and their connectivity

    #init a disctionary to store the bonds present
    bond_dict = dict()

    #get all pairs of interacting particle types that are bonded. log the pairwise body
    #bond matrix in the bond_dict
    for bond_type in sim.bonds:

        #get the types and cutoff distance for that pair
        type_p = bond_type[0]
        type_q = bond_type[1]
        pq_cutoff = bond_type[2] * sim.cutoff_mult

        #update the bond_dict with bodies interacting through these types
        get_p_q_bonds(particle_info, type_p, type_q, bond_dict, sim.box_dim, pq_cutoff,
                      radius, center)



    #determine groups of bonded structures
    G, bond_dict = get_groups(bond_dict)

    #count the sizes of each group
    size_counts, largest_group_size = get_group_sizes(G)

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
    print(frames)

    #gather all the relevant global info into a SimInfo object
    sim = SimInfo(snap, frames, ixn_file = ixn_file)

    #init outfile to dump results
    if write_output:
        fout = open("analysis_out.dat", 'w') 

    #loop over each frame and perform the analysis
    for frame in range(0, frames, jump):

        #get the snapshot for the current frame
        snap = snaps.read_frame(frame)

        #get the particle info for the current frame
        particle_info = get_particles(snap)

        #check if there are nanoparticles in the simulation. If so, get locations
        if (sim.nano_flag):

            nanoparticles = body.get_nanoparticles(particle_info, sim)

        sys.exit()
            


        #analyze the structures based on nanoparticle presence
        if (not sim.nano_flag):
            q = analyze_structures(particle_info, sim)

        else:
            q = []
            for nano_type in range(sim.num_nano_types):
                for nanoparticle in range(len(nano_centers[nano_type])):
                    r      = nano_radii[nano_type] * sim.radius_mult + sim.largest_bond_distance
                    # print(r)
                    center = nano_centers[nano_type][nanoparticle]
                    q_i    = analyze_structures(particle_info, sim, r, center) 
                    q.append(q_i) 

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

    #various trajectory types to test on

    # gsd_file = "test_r473_success.gsd"
    # ixn_file = "../interactionsT3.txt"

    # gsd_file = "../../T3_diamonds/DBOND-5.0_DIMERS-160_NBOND-0.6_2/sd1296.gsd"
    # gsd_file = "/home/anthony/storage/Brandeis/T3_diamonds/naren_test/sd1296.gsd"
    # ixn_file = "../diamond_ixn.txt"

    # gsd_file = "/home/anthony/storage/Brandeis/T3_full/data/e11_8_e23_11_C_1e-4_SEED_1/T3_triangles.gsd"
    # ixn_file = "../interactionsT3.txt"

    # TODO - test on Narens T1 system. Not sure if it is working here for some reason

    # run_analysis(gsd_file, ixn_file, verbose=True, write_output = True)

    #get the name command line arguments - will update to parser in future
    try:
        gsd_file = sys.argv[1]
        ixn_file = sys.argv[2]
        jump     = int(sys.argv[3])
    except:
        print("Usage: %s <gsd_file> <ixn_file> <frame_skip>" % sys.argv[0])
        raise

    #run analysis
    run_analysis(gsd_file, jump = jump, ixn_file = ixn_file, verbose = True, write_output = True)

