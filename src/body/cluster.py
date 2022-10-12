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

    def __init__(self, bodies, frame):

        #create a reference to the list of bodies comprising the cluster
        self.__bodies = bodies

        #init a cluster id to -1
        self.__cluster_index = -1

        #set a last updated value
        self.__last_updated = frame


    def update(self, cluster):
        #update a cluster with up to date member bodies

        self.__bodies = cluster.get_bodies()
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


    #getter functions

    def get_bodies(self):

        return self.__bodies

    def get_body_ids(self):

        return [bod.get_id() for bod in self.__bodies]

    def get_cluster_id(self):

        return self.__cluster_index

    def get_last_updated(self):

        return self.__last_updated

    def __update_bodies(self, bodies):

        #update the bodies in the cluster to have the set id
        for bod in self.__bodies:
            bod.set_cluster_id(self, self.__cluster_index)












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


def update_live(cluster, live_clusters, old_bodies, frame):
    #use the existing live_clusters to assign an id and update cluster

    #loop over bodies in cluster. get their cluster_id from old_bodies. add to set
    possibleMatches = set()
    for body_id in cluster.get_body_ids():

        if len(old_bodies) > 0:
            old_body = old_bodies[body_id]
            old_cluster = old_body.get_cluster()
            print(body_id, old_cluster)
            if old_cluster is not None:
                possibleMatches.add(old_cluster)

    #break into cases depending on the number of possible matches
    
    if len(possibleMatches) == 0:

        '''
        If there are no possibleMatches, none of the bodies in this cluster were in clusters
        during the previous timestep. Thus this is a new cluster. Create a new cluster whose
        id is one more than the end of live_clusters (todo : this naming may cause issues, fix later?)
        '''

        #get the new cluster id
        if len(live_clusters) == 0:
            new_id = 0
        else:
            new_id = live_clusters[-1].get_cluster_id() + 1

        #update the cluster with the new id, and append to live_clusters
        cluster.set_cluster_id(new_id)
        live_clusters.append(cluster)
        print("created new cluster with id ", new_id)

    elif len(possibleMatches) == 1:

        '''
        If there is exactly one possible match, then all the bodies in this cluster were 
        either in the same cluster, or not part of a cluster during the previous timestep. 
        Several cases are possible.
        1) The structure is exactly the same. 
        2) The structure has gained monomers 
        3) The structure has broken into sub-structures
        '''

        #perform a set difference to determine gain/loss/constant of subunits
        match = list(possibleMatches)[0]
        old_not_new = set(match.get_body_ids()).difference(set(cluster.get_body_ids()))
        new_not_old = set(cluster.get_body_ids()).difference(set(match.get_body_ids()))

        # print("set diffs: ", old_not_new, new_not_old)

        #if two or more particles break off
        if len(old_not_new) > 1:

            #how to tell if breaking off into monomers or clusters?
            #if clusters, they will apear in the list of new clusters

            print('help')

        #if the difference is 1 or 0, same or only lost one monomer
        else:

            match.update(cluster)
            print("matched cluster to existing")

    else:

        '''
        If there is more than one potential match, that means that two or more clusters from the
        previous timestep have coalesced during this frame. If one cluster is bigger than the 
        others, keep that one. 
        May also be able to get here if cluster merge and break at the same time. 
        '''
        
        possible_list = list(possibleMatches)
        for possible_match in possible_list:
            print('hi')

            #do a set difference between cluster and possible
            old_not_new = set(possible_match.get_body_ids()).difference(set(cluster.get_body_ids()))
            new_not_old = set(cluster.get_body_ids()).difference(set(possible_match.get_body_ids()))

            print("set diffs: ", old_not_new, new_not_old)



    return




def track_clustering(snap, sim, frame, live_clusters, old_bodies):
    #compute and update cluster stats from the given frame

    #get a list of bodies to analyze
    bodies = body.create_bodies(snap, sim)

    #init a dictionary to store the bonds present - init with empty lists for each body_id
    bond_dict = dict()
    for bod in bodies:
        bond_dict[bod.get_id()] = []

    #determine the bond network using the list of bodies
    body.get_bonded_bodies(bodies, sim, bond_dict)

    #determine groups of bonded structures
    G = get_groups(bond_dict)

    #for each group, create a cluster
    clusters = []
    for group in G:

        #check for dimers or larger, create a cluster object
        if len(group) > 1:

            #extract the involved bodies from the group and create a cluster
            body_list = [bodies[q] for q in group]
            clusters.append(Cluster(body_list, frame))

    #for each cluster, see if it matches an existing cluster. If not, create new
    for cluster in clusters:
        
        #build an array of possible matching clusters from cluster_id of bodies
        update_live(cluster, live_clusters, old_bodies, frame)

    print(len(live_clusters))
    for c in live_clusters:
        print(frame, c.get_last_updated())





    return bodies
    