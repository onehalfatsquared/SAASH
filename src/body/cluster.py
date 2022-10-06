'''

This file contains class implementations for creating and tracking clusters of bodies 
between simulation frames. 

A cluster is simply an indexed collection of bodies. 

Each cluster that forms in a simulation is given a birth frame. It then tracks statistics 
about the cluster from frame to frame, including the monomer concentration at that point. 
If a cluster dissociates, it is given a death frame, and lifetime. 



'''

import gsd.hoomd
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

import warnings
import sys
import os

from body import neighborgrid as ng
from body import body as body




class Cluster:

    def __init__(self, bodies):

        #create a reference to the list of bodies comprising the cluster
        self.__bodies = bodies





    def get_bodies(self):

        return self.__bodies

    def get_body_ids(self):

        return [bod.get_id() for bod in self.__bodies]












class ClusterInfo:

    def __init__(self, frame):

        self.__birth_frame = frame
        self.__death_frame = -1





    def __set_lifetime(self):

        self.__lifetime = self.__death_frame - self.__birth_frame







































####################################################################
################# Bond Dict -> Graph & Clustering ##################
####################################################################


def get_groups(bond_dict):
    #construct arrays containing groups of all bonded bodies

    #compute total number of bodies as number of keys in the bond_dict
    total_states = len(bond_dict.keys())

    #init an array to store mappings to groups
    to_group = -1 * np.ones(total_states, dtype=int)
    to_group[0] = 0

    #init a list to store the groups
    G = [[0]]

    #init a queue and dump for searching bonds
    queue    = [0]
    searched = []

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
    size_dict = dict()

    #loop over groups and increment the corresponding size index
    for group in G:

        L = len(group)
        if L not in size_dict.keys():
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