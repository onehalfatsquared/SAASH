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

    snaps = gsd.hoomd.open(name="../diamonds_T3/sd1296.gsd", mode="rb")
    snap = snaps.read_frame(0)
    frames = len(snaps)
    print(frames)


    #get the snapshot for the current frame
    snap = snaps.read_frame(frame)

    #get the particle info for the current frame
    particle_info = get_particles(snap)

    print(particle_info)









if __name__ == "__main__":

	get_ex_particle_info(500)