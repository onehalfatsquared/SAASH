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
import pandas as pd

import warnings
import sys
import os

from .structure import body
from .structure import cluster as cluster
from .structure import frame as frame

from .simInfo import *

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




def run_analysis(gsd_file, jump = 1, ixn_file = "interactions.txt", observer = None):
    #get number of monomers and dimers at each frame in the sim

    #get the collection of snapshots and get number of frames
    snaps = gsd.hoomd.open(name=gsd_file, mode="rb")
    snap = snaps.read_frame(0)
    frames = len(snaps)

    #gather all the relevant global info into a SimInfo object
    sim = SimInfo(snap, frames, ixn_file = ixn_file)

    #check for observer. if not found create default observer with a warning
    if observer is None:
        observer = cluster.Observer(gsd_file)
        observer.add_observable('num_bodies')
        print("WARNING: Observer not specified. Using default observer:")
        print("Will track individual clusters and their size as they evolve")

    #init an array to track live clusters
    cluster_info  = []

    #loop over each frame and perform the analysis
    f0 = 1
    old_frame = frame.get_data_from_snap(snaps.read_frame(f0-1), sim, f0-1)
    old_frame.create_first_frame(cluster_info, f0-1, observer)
    mon_fracs = []
    monomer_id_sets = []
    for frame_num in range(f0, frames, jump):

        #get the monomer fraction and ids
        mon_fracs.append(old_frame.get_monomer_fraction())
        monomer_id_sets.append(old_frame.get_monomer_ids())

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

    

    return cluster_info, mon_fracs, monomer_id_sets




# if __name__ == "__main__":

#     #get the name command line arguments - will update to parser in future
#     try:
#         gsd_file = sys.argv[1]
#         ixn_file = sys.argv[2]
#         jump     = int(sys.argv[3])
#     except:
#         print("Usage: %s <gsd_file> <ixn_file> <frame_skip>" % sys.argv[0])
#         raise

#     #run analysis
#     run_analysis(gsd_file, jump = jump, ixn_file = ixn_file, verbose = True, write_output = False)

