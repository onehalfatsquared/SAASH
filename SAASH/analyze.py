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
import sys, select
import os

import pickle
import time

from collections import defaultdict

from .structure import body as body
from .structure import cluster as cluster
from .structure import frame as frame

from .util import observer as obs

from .simInfo import *

####################################################################
################# Output Data Class ################################
####################################################################

class ClusterOut:
    '''Basically just a struct that holds the data that needs to be output from
       a cluster analysis run. 

       Holds a list of all cluster_info objects, as well as a time series of 
       monomer fractions and monomer ids. 
    '''

    def __init__(self, cluster_info, monomer_frac, monomer_ids):

        #init three variables that store the output of a cluster type simulation
        self.cluster_info = cluster_info
        self.monomer_frac = monomer_frac
        self.monomer_ids  = monomer_ids





####################################################################
################# Structure Analysis Functions #####################
####################################################################


def analyze_nano(snap, sim, observer):
    #analyze clusters of subunits and their connectivity

    #get a list of bodies to analyze
    bodies = body.create_bodies(snap, sim)

    #get the nanoparticle locations this frame
    nanoparticles = body.get_nanoparticles(snap, sim)

    #init arrays to store things
    all_q = []

    #loop over each nanoparticle
    for nanoparticle in nanoparticles:

        #get the cutoff radius and center of the nanoparticle
        radius = nanoparticle.get_radius() * sim.radius_mult + sim.largest_bond_distance
        center = nanoparticle.get_position()
        rad_cut = radius * radius

        #filter the bodies that are within the cutoff radius of the nanoparticle
        filtered_bodies = [bod for bod in bodies if bod.is_nearby(center, rad_cut, sim.box_dim)]

        #init a dict to store the bonds - init with empty lists for each body_id
        bond_dict = dict()
        for bod in bodies:
            bond_dict[bod.get_id()] = []

        #determine the bond network using the list of filtered bodies
        body.get_bonded_bodies(filtered_bodies, sim, bond_dict)

        #determine groups of bonded structures
        G = cluster.get_groups(bond_dict)

        #get largest cluster size
        G_lens = [len(G[i]) for i in range(len(G))]
        largest_cluster_size = np.max(G_lens)

        #check if there are no clusters on the nanoparticle
        if largest_cluster_size == 1:
            all_q.append((len(filtered_bodies), 0, 0))
            continue

        #get the number of bonds in the largest cluster
        largest_cluster_id = np.argmax(G_lens)
        largest_cluster    = G[largest_cluster_id]
        bonds = 0
        for particle in largest_cluster:
            bonds += len(bond_dict[particle])

        #get the number adsorbed
        num_adsorbed = len(filtered_bodies)

        all_q.append((num_adsorbed, largest_cluster_size, int(bonds / 2)))

    return all_q


####################################################################
################# Main Drivers for each Run Type ###################
####################################################################

def print_progress(frame_num, observer):
    #print progress updates when the frames is multiples of 10% of the total

    #ectract frame info from the observer
    jump        = observer.get_frame_jump()
    start_frame = observer.get_first_frame()
    final_frame = observer.get_final_frame()

    #determine every 10% of the way there
    thresh = 0.1 * (final_frame - start_frame)
    ratio  = int((frame_num - start_frame)/ thresh)
    prev   = int((frame_num - start_frame - jump) / thresh)

    if ratio != prev:
        print("Analyzed frame {} of {}".format(frame_num, final_frame))
    
    return 

def check_observer(observer, gsd_file, sim):
    #check if there is a supplied observer. if not, init default and give warning

    if observer is None:

        print("\nWARNING: Observer not specified. Using default observer:")
        print("Will track individual clusters and their size as they evolve.")
        print("Press any key to confirm within the next 15 seconds and continue...")

        i, o, e = select.select( [sys.stdin], [], [], 15 )
        if not i:
            raise RuntimeError("Default observer not confirmed. Exiting...")

        observer = obs.Observer(gsd_file=gsd_file)
        observer.init_default_set()

    #check if frame requirements are met for final frame. Set to default if not overwritten
    if observer.get_final_frame() is None or observer.get_final_frame() > sim.frames:

        observer.set_final_frame(sim.frames)


    return observer


def handle_cluster(snaps, frames, sim, observer, jump = 1):
    #analyze according to cluster output. Create cluster info objects

    #init an array to store all cluster info objects
    cluster_info  = []

    #init arrays for the time dependence of monomers
    mon_fracs = []
    monomer_id_sets = []

    #analyze the first frame seperately
    f0 = observer.get_first_frame() + 1
    old_frame = frame.get_data_from_snap(snaps[f0-1], sim, f0-1)
    old_frame.create_first_frame(cluster_info, f0-1, observer)

    print("\nBeginning Cluster Analysis")
    jump = observer.get_frame_jump()

    #loop over each frame and perform the analysis
    for frame_num in range(f0, observer.get_final_frame(), jump):

        print_progress(frame_num, observer)

        #get the monomer fraction and ids from the previous frame
        mon_fracs.append(old_frame.get_monomer_fraction())
        monomer_id_sets.append(old_frame.get_monomer_ids())

        #get the snapshot for the current frame
        snap = snaps[frame_num]

        #do the update for the current frame
        cluster_info, old_frame = cluster.track_clustering(snap, sim, int(frame_num/jump), 
                                                           cluster_info, old_frame,
                                                           observer)

    #create a clusterOut object with the relevant data and return it
    out_data = ClusterOut(cluster_info, mon_fracs, monomer_id_sets)
    return out_data

def write_cluster_output(out_data, observer):
    #output the cluster object data to the file in observer

    outfile = observer.get_outfile()

    with open(outfile, 'wb') as f:
        pickle.dump(out_data, f)
        print("Cluster info pickled into file: {}".format(outfile))

    return


def handle_bulk(snaps, frames, sim, observer):
    #analyze data according to bulk output. Get cluster size distribution

    #check if the user wants details on microstates
    focus_list = observer.get_focus_list()

    #init lists to store the cluster size distribution and the largest cluster
    all_sizes   = []
    all_largest = []
    if focus_list is not None:
        all_focused = []

    print("\nBeginning Bulk Analysis")
    jump = observer.get_frame_jump()

    #loop over each frame and perform the analysis
    for frame_num in range(observer.get_first_frame(), observer.get_final_frame(), jump):

        print_progress(frame_num, observer)

        #get the snapshot for the current frame
        snap = snaps[frame_num]

        #make a Frame object, get the size distribution and append to lists
        fr = frame.get_data_from_snap(snap, sim, frame_num)
        if focus_list is None:

            sizes, largest = fr.get_cluster_size_distribution(observer)

        else:

            sizes, largest, focus = fr.get_cluster_size_distribution(observer)
            all_focused.append(focus)
    
        all_sizes.append(sizes)
        all_largest.append(largest)

    if focus_list is None:
        return all_sizes, all_largest
    
    return all_sizes, all_largest, all_focused


def write_bulk_output(out_data, frames, observer, jump = 1):
    #print the cluster size distribution to file in observer

    #extract the data from out data
    cluster_sizes = out_data[0]
    largest_sizes = out_data[1]
    max_size      = max(largest_sizes)

    #get the output file name from observer and open it for writing
    outfile = observer.get_outfile()
    fout = open(outfile, 'w') 

    #write the data
    jump = observer.get_frame_jump()
    frame_counter = 0
    for frame_num in range(observer.get_first_frame(), observer.get_final_frame(), jump):

        #convert dict to cluster array
        cluster_size_array = [cluster_sizes[frame_counter][i] for i in range(max_size+1)] 

        #write output
        fout.write("{} ".format(frame_num))
        for i in range(max_size):
            fout.write("{} ".format(cluster_size_array[i+1]))
        fout.write("{}".format(largest_sizes[frame_counter]))
        fout.write("\n")

        frame_counter += 1

    #close the file
    fout.close()
    print("Cluster sizes written to file: {}".format(outfile))

    #check for focused sizes
    focus_list = observer.get_focus_list()
    if focus_list is not None:

        for csize in focus_list:

            focus_outfile = outfile.split('.sizes')[0] + "_" + str(csize) + ".fsizes"
            write_focus_output(csize, out_data[2], focus_outfile, observer)


    return



def write_focus_output(csize, all_focus, outfile, observer):
    #print a separate file per cluster size, containing info on microstate distribution 

    #open outfile
    fout = open(outfile, 'w') 

    #print the first line, giving the definitions of each microstate
    fout.write("Microstates: ")

    #do an initial pass through all time points to gather all microstates (as keys)
    key_map = defaultdict(int)
    for distribution in all_focus:

        size_dist = distribution[csize]
        for microstate in size_dist:

            if microstate not in key_map:

                key_map[microstate] = len(key_map) 
                fout.write("( {} : {} ),".format(key_map[microstate], microstate))

    fout.write("\n")

    #check if the requested size has no members. if not, delete the file with message to user
    if len(key_map) == 0:

        fout.close()
        os.remove(outfile)
        print("No clusters of size {} found\n".format(csize))
        return

    #write the data
    jump = observer.get_frame_jump()
    frame_counter = 0
    for frame_num in range(observer.get_first_frame(), observer.get_final_frame(), jump):

        distribution = all_focus[frame_counter]
        size_dist    = distribution[csize]

        counts = [0 for i in range(len(key_map))]

        for microstate in size_dist:

            counts[key_map[microstate]] = size_dist[microstate]

        #write all of them in indexed order
        fout.write("{} ".format(frame_num))
        for i in range(len(counts)):

            fout.write("{} ".format(counts[i]))

        fout.write("\n")

        frame_counter += 1

    #close the file
    fout.close()
    print("Cluster microstates (size {}) written to file: {}".format(csize, outfile))
    return


def handle_nanoparticle(snaps, frames, sim, observer):
    #analyze data according to nanoparticle output. 
    #get info about assembly around the nanoparticle

    if not sim.nano_flag:
        error_msg = """Observer has started a nanoparticle analysis, but no 
                       nanoparticle was found in the simulation results. \n
                       Verify that the correct gsd file was used and that the 
                       interaction file contains the nanoparticle definition.\n"
        """
        raise RuntimeError(error_msg)

    nano_data = []

    print("\nBeginning Nanoparticle Analysis")
    jump = observer.get_frame_jump()

    #loop over each frame and perform the analysis
    for frame_num in range(observer.get_first_frame(), observer.get_final_frame(), jump):

        print_progress(frame_num, observer)

        #get the snapshot for the current frame
        snap = snaps[frame_num]

        #get the cluster data for each nanpoparticle
        q = analyze_nano(snap, sim, observer)
        nano_data.append(q)


    return nano_data

def write_nanoparticle_output(out_data, frames, observer):
    #print cluster info around each nanoparticle

    #get the output file name from observer and open it for writing
    outfile = observer.get_outfile()
    fout = open(outfile, 'w') 

    num_nanos = len(out_data[0])
    jump = observer.get_frame_jump()

    frame_counter = 0
    for frame_num in range(observer.get_first_frame(), observer.get_final_frame(), jump):

        #grab this frames data
        q = out_data[frame_counter]

        #print the data to file
        fout.write("{}".format(frame_num))
        for nano in range(num_nanos):
            fout.write(",%s,%s,%s"%(q[nano][0], q[nano][1], q[nano][2]))
        fout.write("\n")

        frame_counter += 1

    #close the file
    fout.close()
    print("Nanoparticle assembly info written to file: {}".format(outfile))

    return







def run_analysis(gsd_file, ixn_file = "interactions.txt", observer = None):
    #get number of monomers and dimers at each frame in the sim

    #get the collection of snapshots and get number of frames
    snaps = gsd.hoomd.open(name=gsd_file, mode="rb")
    snap = snaps[0]
    frames = len(snaps)

    #gather all the relevant global info into a SimInfo object

    #check observer for optional parameters to simInfo
    if observer is not None and observer.get_ngrid_cutoff() is not None:
        
        sim = SimInfo(snap, frames, ixn_file = ixn_file, 
                      ngrid_R = observer.get_ngrid_cutoff())
    else:

        sim = SimInfo(snap, frames, ixn_file = ixn_file)

    #check for observer. if not found create default observer with a warning
    observer = check_observer(observer, gsd_file, sim)

    #get run type info from the observer
    run_type = observer.get_run_type()

    #fork the analysis base don the run type
    if run_type == 'cluster':

        out_data = handle_cluster(snaps, frames, sim, observer)
        write_cluster_output(out_data, observer)

    elif run_type == 'bulk':

        out_data = handle_bulk(snaps, frames, sim, observer)
        write_bulk_output(out_data, frames, observer)

    elif run_type == 'nanoparticle':

        out_data = handle_nanoparticle(snaps, frames, sim, observer)
        write_nanoparticle_output(out_data, frames, observer)


    return 0