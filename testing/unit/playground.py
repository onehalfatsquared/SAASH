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

from body import body
from body import neighborgrid
import analyzeStructures_refactor as test


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









if __name__ == "__main__":

    #dimaond test
    gsd_file = "../diamonds_T3/sd1296.gsd"
    ixn_file = "../diamonds_T3/diamond_ixn.txt"

    #t3 triangle test
    gsd_file = "../triangles_T3/nano_test.gsd"
    ixn_file = "../triangles_T3/interactionsT3.txt"

    #patchy 2d test
    # gsd_file = "../patchy_2d/traj.gsd"
    # ixn_file = "../patchy_2d/interactions.txt"




    # get_ex_particle_info(gsd_file, ixn_file, 500)
    # test_distance()
    # test_subunit_size(gsd_file, ixn_file, 5)
    test.run_analysis(gsd_file, ixn_file = ixn_file)
