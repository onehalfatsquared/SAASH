'''

This file contains class implementations for creating and tracking clusters of bodies 
between simulation frames. 

A cluster is simply an indexed collection of at least two bodies. 

For each cluster that forms, a ClusterInfo object is created. This keeps track of birth
and death frames of a cluster and keeps a time series of cluster properties that the user 
can choose from. Among these is the monomer fraction, which is always tracked, in order
to build statistics on transitions as a function of monomer concentration. 

There is also an observer class that can be set with various quantities to compute for 
each cluster. Examples include number of bodies, number of bonds, positions of bodies, 
a canonical graph labeling, etc. The ClusterInfo takes the observer and reads the 
requested quantities. The ClusterInfo time series then contains a dictionary with these 
values at every logged frame. 

'''

import gsd.hoomd
import numpy as np
import pandas as pd

import warnings
import sys
import os

from collections import defaultdict

from . import body as body
from . import frame as frame

#append parent directory to import util
from inspect import getsourcefile

current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)

from util import neighborgrid as ng

sys.path.pop(0)


class Cluster:

    def __init__(self, bodies, frame_num):

        #create a reference to the list of bodies comprising the cluster
        self.__bodies = bodies

        #init a cluster id to -1
        self.__cluster_index = -1

        #set a last updated value
        self.__last_updated = frame_num


    def get_possible_matches(self, old_bodies):
        #determine which previous clusters each body in current cluster was part of

        #init a set to store possible cluster matches
        possibleMatches = set()

        #loop over bodies in cluster. get their cluster_id from old_bodies. add to set
        for body_id in self.get_body_ids():

            old_body    = old_bodies[body_id]
            old_cluster = old_body.get_cluster()
            # print(body_id, old_cluster)
            if old_cluster is not None:
                possibleMatches.add(old_cluster)

        return possibleMatches

    def get_difference(self, cluster):
        #determine which bodies are in self but not cluster

        my_bodies    = set(self.get_body_ids())
        their_bodies = set(cluster.get_body_ids())

        return my_bodies.difference(their_bodies)


    def update(self, cluster):
        #update a cluster with up to date member bodies

        #set the body list to match given cluster. 
        self.__bodies = cluster.get_bodies()

        #Update those bodies with this cluster id
        self.__update_bodies(self.__bodies)

        #update the frame of the last update
        self.__last_updated = cluster.get_last_updated()


    #setter functions

    def set_cluster_id(self, c_id):
        #set the given id to the cluster
        #also sets that id to each body, and gives the bodies a ref to cluster

        self.__cluster_index = c_id
        self.__update_bodies(self.__bodies)

        return

    def steal_id(self, victim):
        #take the cluster id from victim cluster, set it for yourself, leave victim with -1

        self.set_cluster_id(victim.get_cluster_id())
        victim.set_cluster_id(-1)

        return


    #getter functions

    def get_bodies(self):

        return self.__bodies

    def get_num_bodies(self):

        return len(self.__bodies)

    def get_body_ids(self):

        return [bod.get_id() for bod in self.__bodies]

    def get_body_positions(self):

        return [bod.get_position() for bod in self.__bodies]

    def get_cluster_id(self):

        return self.__cluster_index

    def get_last_updated(self):

        return self.__last_updated

    def __update_bodies(self, bodies):

        #update the bodies in the cluster to have the set id
        for bod in self.__bodies:
            bod.set_cluster_id(self, self.__cluster_index)




class ClusterInfo:

    def __init__(self, cluster, frame_num, monomer_frac, observer):

        #init information about birth and death frame for this cluster
        self.__birth_frame  = frame_num
        self.__death_frame  = -1
        self.__last_updated = -1
        self.__lifetime     = -1
        self.__is_dead      = False
        self.__is_absorbed  = False
        self.__has_parent   = False

        #init an observer
        self.__observer = observer

        #init storage for observed variables
        self.__stored_data = []
        self.update_data(cluster, frame_num, monomer_frac)

        #init storage for monomer stats
        self.__from_monomer = dict()
        self.__to_monomer   = dict()


        #todo - check that timescale gives number of entries in data. 
        #may need to offset this by 1, since counting starts at 0


    def get_transitions(self, t0, lag):
        #return a list of all transitions that occur between t0 and lag

        #init event list
        events = []

        #start with the obvious transition in stored data
        start_data = self.__stored_data[t0]

        #only do end data if within the lifetime of the cluster
        if (self.is_dead() and not self.has_parent() and t0+lag+self.__birth_frame == self.__death_frame):
            pass

        else:

            if (t0+lag < len(self.__stored_data)):
                end_data   = self.__stored_data[t0+lag]

                #todo - add in a state conversion fn handle, for now just use num_bodies
                events.append((start_data['num_bodies'], end_data['num_bodies']))

        #perform dictionary comps to determine which events happened in this lag
        added_mons = {key:value for key,value in self.__from_monomer.items() if (t0 < key-self.__birth_frame <= t0+lag)}
        lost_mons  = {key:value for key,value in self.__to_monomer.items() if (t0 < key-self.__birth_frame <= t0+lag)}

        #if these compressed dicts are non-empty

        #addition events
        for key in added_mons.keys():

            #get the number of added monomers
            num_added = added_mons[key]['num_monomers']
            for i in range(num_added):

                events.append((1,end_data['num_bodies']))

        #subtraction events
        for key in lost_mons.keys():

            #get the number of lost monomers
            num_lost = lost_mons[key]['num_monomers']
            for i in range(num_lost):

                events.append((start_data['num_bodies'],1))

        #if looking at first frame, add in monomerization events. skip if splitting from a parent
        if t0 == 0 and not self.__has_parent:

            #get the number of added monomers - cant add more than what we end with
            num_added = self.__from_monomer[self.__birth_frame]['num_monomers']

            if (self.is_dead() and t0+lag+self.__birth_frame == self.__death_frame):
                end_data = self.__stored_data[t0+lag-1]
            elif len(self.__stored_data) == 1:
                end_data = self.__stored_data[t0]
            else:
                end_data = self.__stored_data[t0+lag-1]

            upper_bound = end_data['num_bodies']

            for i in range(min(num_added,upper_bound)):

                events.append((1,end_data['num_bodies']))

        return events



    def set_parent(self, cluster, prev_monomer):
        #set the given cluster to be the parent of the cluster, i.e. first in stored data

        if not self.__has_parent:
            self.__stored_data.insert(0, self.__compute_coordinate(cluster))
            self.__stored_data[0]['monomer_fraction'] = prev_monomer
            self.__has_parent = True

        return

    def kill(self, frame_num):
        # set this cluster to dead status

        self.__death_frame = frame_num
        self.__set_lifetime()
        self.__is_dead = True

        return

    def absorb(self, frame_num):
        # set this cluster to dead status

        self.__death_frame = frame_num
        self.__set_lifetime()
        self.__is_absorbed = True

        return

    def update_data(self, cluster, frame_num, monomer_frac):
        #append the current cluster's coordinate data to storage

        if (frame_num > self.__last_updated and not self.__is_dead):

            #compute a dictionary of requested values using this cluster
            self.__stored_data.append(self.__compute_coordinate(cluster))

            #append the monomer fraction when this cluster formed
            self.__stored_data[-1]['monomer_fraction'] = monomer_frac

            #updated the time of last update to current frame
            self.__last_updated = frame_num

        return

    def add_monomers(self, cluster, frame_num, num_monomers, monomer_frac):
        #update the from_monomer list to denote a transition from monomer to cluster

        #compute a dictionary of requested values using this cluster
        self.__from_monomer[frame_num] = self.__compute_coordinate(cluster)

        #append the number of added monomers and fraction from previous timestep
        self.__from_monomer[frame_num]['num_monomers']     = num_monomers
        self.__from_monomer[frame_num]['monomer_fraction'] = monomer_frac

        return

    def remove_monomers(self, cluster, frame_num, num_monomers, monomer_frac):
        #update the to_monomer list to denote a transition from cluster to monomer

        #compute a dictionary of requested values using this cluster
        self.__to_monomer[frame_num] = self.__compute_coordinate(cluster)

        #append the number of added monomers and fraction from previous timestep
        self.__to_monomer[frame_num]['num_monomers']     = num_monomers
        self.__to_monomer[frame_num]['monomer_fraction'] = monomer_frac

        return

    def get_data(self):

        return self.__stored_data

    def get_monomer_gain_data(self):

        return self.__from_monomer

    def get_monomer_loss_data(self):

        return self.__to_monomer

    def get_lifetime(self):

        return self.__lifetime

    def is_dead(self):

        return self.__is_dead

    def is_absorbed(self):

        return self.__is_absorbed

    def has_parent(self):

        return self.__has_parent


    def __set_lifetime(self):

        self.__lifetime = self.__death_frame - self.__birth_frame

        return

    def __compute_coordinate(self, cluster):
        '''Compute the all quantities stored in the observer on the given cluster'''


        return self.__observer.compute_coordinate(cluster)


####################################################################
################# Bond Dict -> Graph & Clustering ##################
####################################################################


def get_groups(bond_dict):
    #construct arrays containing groups of all bonded bodies

    #compute total number of bodies as number of keys in the bond_dict
    total_states = len(bond_dict)

    #init an array to store mappings to groups
    to_group = -1 * np.ones(total_states, dtype=int)
    to_group[0] = 0

    #init a list to store the groups
    G = [[0]]

    #init a queue and dump for searching bonds
    queue    = [0]
    searched = []  #init to flase * N_bod. Index instead of searching this list

    #loop over the queue until it is empty and all bonds have been assigned
    while len(queue) > 0:

        #pop the front of the queue and get its group number (which should be >=0)
        bod = queue.pop(0)
        group_num = to_group[bod]

        #get the adjacency list from bond_dict for this body
        adj_list = bond_dict[bod]

        #for each adjacent body, add it to the group, set the map, add to queue
        for member in adj_list:

            #if the member is not in the group yet, add it and set the mapping
            if member not in G[group_num]:
                G[group_num].append(member)
                to_group[member] = group_num

            #if the member has not been searched and is not in queue, add to queue
            if member not in searched and member not in queue:
                queue.append(member)

        #append the current body to the searched array
        searched.append(bod)

        #check if the queue is empty but there are undetermined states
        if len(queue) == 0 and len(searched) != total_states:

            #determine the first unassigned state
            unassigned = np.where(to_group == -1)[0][0]

            #create a new group for it, assign it, append to queue
            G.append([unassigned])
            to_group[unassigned] = len(G)-1
            queue.append(unassigned)

    #return the list of grouped particles
    return G


def get_group_sizes(G):
    #make a histogram of cluster sizes and determine the largest cluster

    '''init a dictionary to store histogram data
       number of clusters (value) of each size (key)'''
    size_dict = defaultdict(int)

    #loop over groups and increment the corresponding size index
    for group in G:

        L = len(group)
        if L not in size_dict:
            size_dict[L] = 1
        else:
            size_dict[L] += 1

    #determine the largest group
    non_zero_counts = len(size_dict.values())
    if non_zero_counts > 0:
        largest_group_size = np.max(np.array(list(size_dict.keys())))
    else:
        largest_group_size = 0

    #return size counts
    return size_dict, largest_group_size



def track_clustering(snap, sim, frame_num, cluster_info, old_frame, observer):
    #compute and update cluster stats from the given frame

    #determine clusters and init a Frame object with the needed data
    current_frame = frame.get_data_from_snap(snap, sim, frame_num)

    #map the clusters from previous timestep to this step to assign labels
    # print("Frame {} updates:".format(frame_num))
    current_frame.update(cluster_info, old_frame, observer)

    #return updated cluster info and the bodies to be used as old_bodies next frame
    return cluster_info, current_frame