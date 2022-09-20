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
import analyzeStructures_refactor as test


def get_ex_particle_info(gsd_file, frame):

    snaps = gsd.hoomd.open(name=gsd_file, mode="rb")
    snap = snaps.read_frame(0)
    frames = len(snaps)
    print(frames)

    #get the snapshot for the current frame
    snap = snaps.read_frame(frame)

    #get siminfo
    sim = test.SimInfo(snap, frames, ixn_file = ixn_file)

    #get the particle info for the current frame
    particle_info = test.get_particles(snap)

    return particle_info, sim



def test_nano_create(particle_info, sim):

    nanoparticles = body.get_nanoparticles(particle_info, sim)
    for i in range(len(nanoparticles)):
        nt = nanoparticles[i].get_type()
        nr = nanoparticles[i].get_radius()
        np = nanoparticles[i].get_position()

        print("Nanoparticle of type {} and radius {} at location {}".format(nt, nr, np))





if __name__ == "__main__":

    #set file and frame to look at 
    gsd_file = "sd1296.gsd"
    ixn_file = "diamond_ixn.txt"
    frame = 500

    #get siminfo and particle info
    particle_info, sim = get_ex_particle_info(gsd_file, 500)

    test_nano_create(particle_info, sim)