'''

This is for dong various experiments to help with implemnting features

'''
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




def get_ex_particle_info(frame):

    #open file
    snaps = gsd.hoomd.open(name="../diamonds_T3/sd1296.gsd", mode="rb")
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
    # print(snap.particles.typeid)
    # mask = [i for i,x in enumerate(snap.particles.typeid) if x in sim.interacting_types_mapped]
    # print(snap.particles.position[mask])
    # print(snap.particles.body[mask])

    #test creating bodies from the snap
    bodies = body.create_bodies(snap, sim)
    print(bodies[77].get_particles()[1].get_type())









if __name__ == "__main__":

    gsd_file = "../diamonds_T3/sd1296.gsd"
    ixn_file = "../diamonds_T3/diamond_ixn.txt"
    # get_ex_particle_info(500)

    test.run_analysis(gsd_file, ixn_file = ixn_file)