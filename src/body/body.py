'''

This file contains the class implemntations for bodies, particles, and bonds. 

'''

import gsd.hoomd
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

import warnings
import sys
import os


class Nano:

    def __init__(self, ntype, radius, position = None):

        #set the variables given that define a nanoparticle
        self.__position = position
        self.__ntype    = ntype
        self.__radius   = radius

    #getter functions 

    def get_position(self):

        return self.__position

    def get_type(self):

        return self.__ntype

    def get_radius(self):

        return self.__radius
        

class Bond:

    def __init__(self, type1, type2, cutoff):

        #set the variables given that define the bond
        self.__type1  = type1
        self.__type2  = type2
        self.__cutoff = cutoff

        #give the bond a descriptive strign name - "type1-type2"
        self.__bond_name = type1 + "-" + type2

    #getter functions

    def get_types(self):

        return tuple((self.__type1, self.__type2))

    def get_cutoff(self):

        return self.__cutoff

    def get_name(self):

        return self.__bond_name



class Particle:

    def __init__(self, position, p_type, body):

        #set the variables given in constructor
        self.__position = position
        self.__p_type   = p_type
        self.__body     = body

        #get the body id from the body object
        self.__body_id  = body.get_id()

    #getter functions 

    def get_position(self):

        return self.__position

    def get_type(self):

        return self.__p_type

    def get_body_id(self):

        return self.__body_id

    def get_body(self):

        return self.__body



class Body:

    def __init__(self, particle_pos, particle_type, body_index):

        #set the body index - corresponds to placement in the array of bodies
        self.__body_index = body_index

        #init a center of mass variable for the body
        center_mass = particle_pos[0] * 0

        #init an array for particle objects. 
        self.__num_particles = len(particle_pos)
        self.__particles = []

        #create the particle objects and append to array
        for i in range(self.__num_particles):

            #create and append particle
            particle = Particle(particle_pos[i], particle_type[i], self)
            self.__particles.append(particle)

            #update the center of mass
            center_mass += particle_pos[i]

        #divide center of mass by num particles and set it
        self.__center_mass = center_mass / self.__num_particles

    #setter functions

    def set_position(self, position):
        #manually set position of the body

        self.__position = position

    #getter functions

    def get_position(self):

        return self.__center_mass

    def get_id(self):

        return self.__body_index

    def get_num_particles(self):

        return self.__num_particles

    def get_particles(self):

        return self.__particles









def create_bodies(snap, sim):
    #create an array of all bodies containing particles relevant to the assembly process

    #init a list to store the bodies
    bodies = []

    #get the list of relevant particle types - hoomd index
    interacting_types = sim.interacting_types_mapped

    #create a mask for accessing particle position data, get positions+body of these types
    mask = [i for i,x in enumerate(snap.particles.typeid) if x in interacting_types]
    filtered_pos   = snap.particles.position[mask]
    filtered_bod   = snap.particles.body[mask]
    filtered_types = snap.particles.typeid[mask]

    #map the filtered_types from hoomd ints back into text strings for particle creation
    filtered_types = np.array([snap.particles.types[element] for element in filtered_types])

    #create a set of the masked bodies to get the unique ids
    unique_bods = set(filtered_bod)

    #loop over the masked bodies, creating a Body object with the relevant particles
    body_index = 0    #init a counter so body indexing starts at 0
    for body_id in unique_bods:

        #create a sub-mask to get only particles part of the current body
        sub_mask = [i for i,x in enumerate(filtered_bod) if x == body_id]

        #get the positions and types or particles at the indices set by this submask
        particle_positions = filtered_pos[sub_mask]
        particle_types     = filtered_types[sub_mask]

        #create the body, append it to the list
        current_body = Body(particle_positions, particle_types, body_index)
        bodies.append(current_body)

        #increment body counter
        body_index += 1

    #return the list of bodies
    return bodies
        


def get_particle_info(snap):
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




def get_nano_centers(particle_info, particle_type):
    #get the list of coordinates for all nanoparticle centers

    #get the list of all particle info for that type
    center_list = particle_info.loc[(particle_info['type'] == particle_type)]

    #extract the coordinates from this list of data
    c_coords = np.array([np.array(center_list['position_x'].values), 
                         np.array(center_list['position_y'].values), 
                         np.array(center_list['position_z'].values)]).T

    #return the coordinates
    return c_coords



def get_nanoparticles(particle_info, sim):
    #construct data structures for all nanoparticles in the system

    #init array to store all nanoparticles
    nanoparticles = []

    #loop over each type, appending coordinates and radii to the list
    for i in range(sim.num_nano_types):

        #grab the type and radius from the abstract nanoparticle list
        nano_type = sim.nanos[i].get_type()
        nano_rad  = sim.nanos[i].get_radius()

        #get the coordinates of all nanoparticles of this type
        nano_coords = get_nano_centers(particle_info, nano_type)

        #for each, construct a nanoparticle object and append it to an array
        for coordinates in nano_coords:
            nanoparticle = Nano(nano_type, nano_rad, coordinates)
            nanoparticles.append(nanoparticle)

    #return the array of nanoparticle objects
    return nanoparticles


