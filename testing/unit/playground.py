'''

This is for dong various experiments to help with implemnting features

'''
import hoomd
import gsd.hoomd
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

import warnings

import os
import pytest

#add the paths to the relevant folder for testing
import sys
sys.path.insert(0, '../../src')

#import profiling tools
import cProfile, pstats
import re
import time

from body import body
from body import neighborgrid
from body import cluster
from body import frame
import analyzeStructures_refactor as test
import analyzeStructures as original


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


def test_distance():
    #test if periodic BC works as intended for distance function

    box = np.array([5,5])

    x1 = np.array([4,2])
    x2 = np.array([1,2])

    print(distance(x1,x2,box))

    return

def test_subunit_size(gsd_file, ixn_file, frame):
    #try to compute a size of a subunit using particle data

    snaps = gsd.hoomd.open(name=gsd_file, mode="rb")
    snap = snaps.read_frame(frame)
    box = snap.configuration.box
    box_dim = np.array(box[0:3])
    frames = len(snaps)
    print(frames)

    for i in range(1200):
        print(i, snap.particles.types[snap.particles.typeid[i]])
    sys.exit()

    sim = test.SimInfo(snap, frames, ixn_file = ixn_file)

    #filter by body
    mask = [i for i,x in enumerate(snap.particles.body) if x == 11]
    pos = snap.particles.position[mask]
    masked_pos = snap.particles.position[mask]
    masked_types = snap.particles.typeid[mask]
    sub_mask = [i for i,x in enumerate(masked_types) if x in sim.interacting_types_mapped]
    double_masked_pos = masked_pos[sub_mask]
    double_masked_types = masked_types[sub_mask]


    center = pos[0]
    for i in range(1,len(double_masked_pos)):

        print(i, double_masked_types[i], distance(center, double_masked_pos[i], box_dim))




def get_ex_particle_info(gsd_file, ixn_file, frame):

    #open file
    snaps = gsd.hoomd.open(name=gsd_file, mode="rb")
    snap = snaps.read_frame(0)
    frames = len(snaps)
    print(frames)

    #create the simInfo
    sim = test.SimInfo(snap, frames, ixn_file = ixn_file)
    print(sim.type_map)
    print(sim.interacting_types_mapped)

    #get the snapshot for the current frame
    snap = snaps.read_frame(frame)

    #do tests on extracting types and grabbing fields corresponding to given types
    print(snap.particles.typeid)
    print(snap.particles.types)
    mask = [i for i,x in enumerate(snap.particles.typeid) if x in sim.interacting_types_mapped]
    print(snap.particles.position[mask])
    print(snap.particles.body[mask])


    sub_mask = [i for i,x in enumerate(snap.particles.body) if x == 11]
    print(snap.particles.typeid[sub_mask])

    #test creating bodies from the snap
    bodies = body.create_bodies(snap, sim)
    print(bodies[-1].get_particles()[1].get_type())
    print(bodies[-1].get_particles_by_type("A5A"))

    return

def plot_clusters(sim, coordinates):
    #coordinates contains time series of cluster body positions. plot them
     
    figure, ax = plt.subplots()
    box_dim = sim.box_dim
    ax.set_xlim((-box_dim[0]/2, box_dim[0]/2))
    ax.set_ylim((-box_dim[1]/2, box_dim[1]/2))

    # Loop
    for i in range(len(coordinates)):

        ax.patches = []
        bodies = coordinates[i]
        for bod in bodies:
            center = (bod[0], bod[1])
            circle = plt.Circle(center, 0.5, color='blue')
            ax.add_patch(circle)

        print(i)
     

        plt.pause(0.1)


def run_and_plot_clusters():

    #patchy 2d test
    # gsd_file = "../patchy_2d/traj.gsd"
    gsd_file = "../patchy_2d/traj_multi.gsd"
    ixn_file = "../patchy_2d/interactions.txt"
    jump = 1

    #get the collection of snapshots and get number of frames
    snaps = gsd.hoomd.open(name=gsd_file, mode="rb")
    snap = snaps.read_frame(0)
    frames = len(snaps)

    #gather all the relevant global info into a SimInfo object
    sim = test.SimInfo(snap, frames, ixn_file = ixn_file)

    #create an observer to compute requested observables
    observer = cluster.Observer(gsd_file)
    observer.init_test_set()

    #init an array to track live clusters
    cluster_info  = []
    old_bodies    = []

    f0 = 500
    old_frame = frame.get_data_from_snap(snaps.read_frame(f0), sim, f0)
    old_frame.create_first_frame(cluster_info, f0, observer)

    #loop over each frame and perform the analysis
    max_frame = f0+100
    for frame_num in range(f0, max_frame, jump):

        #print message to user about frame num
        print("Analyzing frame ", frame_num)

        #get the snapshot for the current frame
        snap = snaps.read_frame(frame_num)

        #do the cluster tracking
        cluster_info, old_frame = cluster.track_clustering(snap, sim, frame_num, 
                                                           cluster_info, old_frame,
                                                           observer)

    #grab coordinates from a test cluster to plot in time
    longest_traj = 0
    longest_id   = -1
    for i in range(len(cluster_info)):
        if len(cluster_info[i].get_data()) > longest_traj:
            longest_traj = len(cluster_info[i].get_data())
            longest_id = i

    test_cluster = longest_id
    cluster_data = cluster_info[test_cluster].get_data()
    test_coordinates = []
    for i in range(len(cluster_data)):
        test_coordinates.append(cluster_data[i]['positions'])

    plot_clusters(sim, test_coordinates)




    return 




def run_profile():

    #dimaond test
    gsd_file = "../diamonds_T3/sd1296.gsd"
    ixn_file = "../diamonds_T3/diamond_ixn.txt"

    #t3 triangle test
    # gsd_file = "../triangles_T3/nano_test.gsd"
    # ixn_file = "../triangles_T3/interactionsT3.txt"

    #patchy 2d test
    gsd_file = "../patchy_2d/traj.gsd"
    ixn_file = "../patchy_2d/interactions.txt"

    test.run_analysis(gsd_file, ixn_file = ixn_file, jump=50)
    # original.run_analysis(gsd_file, ixn_file = ixn_file, jump = 50)



if __name__ == "__main__":

    #dimaond test
    gsd_file = "../diamonds_T3/sd1296.gsd"
    ixn_file = "../diamonds_T3/diamond_ixn.txt"

    #t3 triangle test
    # gsd_file = "../triangles_T3/nano_test.gsd"
    # ixn_file = "../triangles_T3/interactionsT3.txt"

    #patchy 2d test
    # gsd_file = "../patchy_2d/traj.gsd"
    gsd_file = "../patchy_2d/traj_multi.gsd"
    ixn_file = "../patchy_2d/interactions.txt"



    #random testing stuff
    # get_ex_particle_info(gsd_file, ixn_file, 500)
    # test_distance()
    # test_subunit_size(gsd_file, ixn_file, 5)
    # test.run_analysis(gsd_file, ixn_file = ixn_file)

    #plot clustering test
    run_and_plot_clusters()



    #profiling for speed
    # cProfile.run('run_profile()', 'restats')
    # p = pstats.Stats('restats')
    # p.strip_dirs().sort_stats('tottime').print_stats(15)
