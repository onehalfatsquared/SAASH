'''

This file contains the class implementations for bodies, particles, bonds, and 
nanoparticles. 

Also contains utitility functions for doing computations on these objects.  

'''

import gsd.hoomd
import numpy as np
import pandas as pd

import warnings
import sys
import os

#append parent directory to import util
from inspect import getsourcefile

current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)

from util import neighborgrid as ng

sys.path.pop(0)


####################################################################
################# Class Implementations  ###########################
####################################################################


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


    def is_bonded(self, particle, cutoff, box):
        #determine if this particle is bonded to the given particle

        #get the periodic distance between the particles
        particle_distance = distance(self.__position, particle.get_position(), box)

        #compare to given cutoff
        if (particle_distance < cutoff):
            return True
        
        return False

    def bind(self, particle, bond_dict = None):
        #bind the two particles together by augmenting their bodies bond list

        #get the host and target particle body
        target_body = particle.get_body()
        host_body   = self.__body

        #get the id of each of these bodies
        target_body_id = target_body.get_id()
        host_body_id   = host_body.get_id()

        #bind the host particle body to the given particle body
        if not host_body.is_bonded(target_body):

            #bind on the body level
            host_body.bind(target_body)

            #add indices to bond dict
            if (bond_dict is not None):
                bond_dict[host_body_id].append(target_body_id)

        #bind the given body to the host particle
        if not target_body.is_bonded(host_body):

            #bind on the body level
            target_body.bind(host_body)

            #add indices to bond dict
            if (bond_dict is not None):
                bond_dict[target_body_id].append(host_body_id)


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

        #set a cluster index, init to -1
        self.__cluster       = None
        self.__cluster_index = -1

        #init a list to store bonds - bodies in this list are bound
        self.__bond_list = []

        #init a size, type, and position for the body
        self.__position  = particle_pos[0] * 0
        self.__body_type = ""
        self.__size      = 0

        #init an array for particle objects. 
        self.__num_particles = len(particle_pos)
        self.__particles = []
        self.__particle_type_map = dict()

        #create the particle objects and append to array
        for i in range(self.__num_particles):

            #create and append particle
            particle = Particle(particle_pos[i], particle_type[i], self)
            self.__particles.append(particle)

            #add this particle ID to the type dictionary
            if (particle_type[i] in self.__particle_type_map.keys()):

                self.__particle_type_map[particle_type[i]].append(i)
            else:

                #the type is not yet a key, so associate an empty list and append this index
                self.__particle_type_map[particle_type[i]] = []
                self.__particle_type_map[particle_type[i]].append(i)


    #bonding related function

    def is_bonded(self, body):
        #determine if the host body is already bonded to the given one

        if body in self.__bond_list:
            return True

        return False

    def is_nearby(self, position, cutoff, box):
        #return true if particle is within cutoff of given position

        #get the periodic distance between the body and pos
        dist = distance(self.__position, position, box)

        #compare to given cutoff
        if (particle_distance < cutoff):
            return True
        
        return False

    def bind(self, body):
        #append the body to the hosts bond list

        self.__bond_list.append(body)

    #setter functions

    def set_position(self, position):
        #manually set position of the body

        self.__position = position

    def set_type(self, body_type):
        #manually set the type for the body

        self.__body_type = body_type

    def set_cluster_id(self, cluster, c_id):

        self.__cluster = cluster
        self.__cluster_index = c_id


    #getter functions

    def get_position(self):

        return self.__position

    def get_id(self):

        return self.__body_index

    def get_cluster(self):

        return self.__cluster

    def get_cluster_id(self):

        return self.__cluster_index

    def get_type(self):

        return self.__body_type

    def get_num_particles(self):

        return self.__num_particles

    def get_particles(self):

        return self.__particles

    def get_particles_by_type(self, particle_type):

        return [particle for particle in self.__particles if particle.get_type() == particle_type]

    def get_bond_list(self):

        return self.__bond_list

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


def get_nano_centers(snap, sim, particle_type):
    #get the coordinates for the center of each nanoparticle of the specified type

    #get the hoomd int index corresponding to the string particle type
    hoomd_nano_type = sim.type_map[particle_type]

    #create a mask to get indices for all nanoparticles of the given type
    mask = np.where(snap.particles.typeid == hoomd_nano_type)[0]
    filtered_pos   = snap.particles.position[mask]

    #check if problem is 2d or 3d. Cut out z component if 2d
    if (sim.dim == 2):
        filtered_pos = filtered_pos[:,0:2]

    return filtered_pos



def get_nanoparticles(snap, sim):
    #construct data structures for all nanoparticles in the system

    #init array to store all nanoparticles
    nanoparticles = []

    #loop over each type, appending coordinates and radii to the list
    for i in range(sim.num_nano_types):

        #grab the type and radius from the abstract nanoparticle list
        nano_type = sim.nanos[i].get_type()
        nano_rad  = sim.nanos[i].get_radius()

        #get the coordinates of all nanoparticles of this type
        nano_coords = get_nano_centers(snap, sim, nano_type)

        #for each, construct a nanoparticle object and append it to an array
        for coordinates in nano_coords:
            nanoparticle = Nano(nano_type, nano_rad, coordinates)
            nanoparticles.append(nanoparticle)

    #return the array of nanoparticle objects
    return nanoparticles


####################################################################
############ Body Creation and Bond Network Detection ##############
####################################################################


def check_particle_pairs(particles1, particles2, cutoff, sim, bond_dict):
    #check if any of the particle1's are within cutoff of particle2's

    #do pairwise comparisons between each particle
    for particle1 in particles1:
        for particle2 in particles2:

            #check if particles are within cutoff. If so, create a bond between bodies
            if (particle1.is_bonded(particle2, cutoff*sim.cutoff_mult, sim.box_dim)):
                particle1.bind(particle2, bond_dict)
                # print("binding {} with {}".format(particle1.get_type(), particle2.get_type()))
                # print(particle1.get_position(), particle2.get_position())
                return True

    #if none of the particles have a bond, return False
    return False



def check_body_pair(body1, body2, sim, bond_dict):
    ''' Check if the pair of bodies contains particles that are within the cutoff of a 
        given bond type. If one is found, we assume the bodies are bonded and stop 
        checking for further bonds
    '''

    #loop over each bond type 
    for bond in sim.bonds:

        #get the two particle types involved in this bond
        type1, type2 = bond.get_types()
        cutoff       = bond.get_cutoff()

        #first get all type 1 on body 1 and type 2 on body 2
        particles1 = body1.get_particles_by_type(type1)
        particles2 = body2.get_particles_by_type(type2)

        #do pairwise comparisons between each particle
        found_bond = check_particle_pairs(particles1, particles2, cutoff, sim, bond_dict)
        if (found_bond):
            return

        #if two particles types are the same, we are done with this bond type
        if type1 == type2:
            continue

        #if the two particles types are different, the reverse check must be performed
        particles1 = body1.get_particles_by_type(type2)
        particles2 = body2.get_particles_by_type(type1)

        #do pairwise comparisons between each particle
        found_bond = check_particle_pairs(particles1, particles2, cutoff, sim, bond_dict)
        if (found_bond):
            return

    return


def get_bonded_bodies(bodies, sim, bond_dict):
    #determine bonded bodies by looping over each neighborhood, bond type, and particle

    #extract and update the neighborgrid using the current bodies info
    ngrid = sim.ngrid
    ngrid.update(bodies)

    #loop over each body
    body = 0
    for current_body in bodies:

        #get all the nearby bodies from the neighborgrid
        nearby_bodies = ngrid.getNeighborhood(current_body)

        # print("Current: ", body, current_body.get_position())
        body += 1

        #loop over nearby bodies, checking for formation of each bond type
        for target_body in nearby_bodies:

            # print(target_body.get_position())

            #check that these bodies are not already bonded. if so, go to next
            if (current_body.is_bonded(target_body)):
                continue

            #check if the two bodies contain bonded particles and update accordingly
            check_body_pair(current_body, target_body, sim, bond_dict)

    return


def get_body_center_dict(snap, sim, unique_bods):
    #for each body we consider, get the name of its center particle and its position
    #return the info as a dictionary

    #init a dict to store the info since the bodies are in a set
    body_info_dict = dict()

    #loop over the first num_bodies + num_nanos elements. These are all centers
    body_count = sim.num_bodies + sim.num_nanos
    for i in range(body_count):

        #get the body id and see if it is in the set we are considering
        body_id = snap.particles.body[i]
        if (body_id) not in unique_bods:
            continue

        #extract position and type of the body
        pos   = snap.particles.position[i]
        p_type = snap.particles.types[snap.particles.typeid[i]]

        # append this info to the dictionary
        body_info_dict[body_id] = [pos, p_type]

    return body_info_dict


def filter_bodies(snap, sim):
    #filter the particle info in the snap to only include bodies involved in interactions

    #get the list of relevant particle types - hoomd index
    interacting_types = sim.interacting_types_mapped

    #create a mask for accessing particle position data, get positions+body of these types
    mask = np.where(np.isin(snap.particles.typeid, interacting_types))[0]
    filtered_pos   = snap.particles.position[mask]
    filtered_bod   = snap.particles.body[mask]
    filtered_types = snap.particles.typeid[mask]

    #check if problem is 2d or 3d. Cut out z component if 2d
    if (sim.dim == 2):
        filtered_pos = filtered_pos[:,0:2]

    #map the filtered_types from hoomd ints back into text strings for particle creation
    filtered_types = np.array([snap.particles.types[element] for element in filtered_types])

    return filtered_pos, filtered_bod, filtered_types


def create_body(filtered_pos, filtered_bod, filtered_types, body_info_dict, body_id, body_index):
    #extract the data for the given body_id and create and return the body

    #create a sub-mask to get only particles part of the current body
    sub_mask = np.where(filtered_bod == body_id)[0]

    #get the positions and types or particles at the indices set by this submask
    particle_positions = filtered_pos[sub_mask]
    particle_types     = filtered_types[sub_mask]

    #create the body, set its position and type from the dict, append it to the list
    current_body = Body(particle_positions, particle_types, body_index)
    current_body.set_position(body_info_dict[body_id][0])
    current_body.set_type(body_info_dict[body_id][1])

    return current_body


def create_bodies(snap, sim):
    #create an array of all bodies containing particles relevant to the assembly process

    #init a list to store the bodies
    bodies = []

    #filter out particle data from snap to only include the bodies relevant to assembly
    filtered_pos, filtered_bod, filtered_types = filter_bodies(snap, sim)

    #create a set of the filtered bodies to get the unique ids
    unique_bods = set(filtered_bod)

    #for each unique body, get its particle type and center of mass in a dictionary
    body_info_dict = get_body_center_dict(snap, sim, unique_bods)

    #loop over the masked bodies, creating a Body object with the relevant particles
    body_index = 0    #init a counter so body indexing starts at 0
    for body_id in unique_bods:

        #create a body for the given id and add it to the list
        current_body = create_body(filtered_pos, filtered_bod, filtered_types, 
                                   body_info_dict, body_id, body_index)

        bodies.append(current_body)

        #increment body counter
        body_index += 1

    #return the list of bodies
    return bodies
        



