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
import matplotlib.pyplot as plt 
import pandas as pd

import warnings
import sys
import os

from body import neighborgrid as ng
from body import body as body


class Frame:

    def __init__(self):

        self.__clusters = []
        self.__bodies   = []

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


    #getter functions

    def get_bodies(self):

        return self.__bodies

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

    def __init__(self, cluster, frame, observer):

        #init information about birth and death frame for this cluster
        self.__birth_frame  = frame
        self.__death_frame  = -1
        self.__last_updated = -1
        self.__lifetime     = -1
        self.__is_dead      = False
        self.__has_parent   = False

        #init an observer
        self.__observer = observer

        #init storage for observed variables
        self.__stored_data = []
        self.update_data(cluster, frame, observer.current_monomer)

        #init storage for monomer stats
        self.__from_monomer = []
        self.__to_monomer   = []


        #todo - check that timescale gives number of entries in data. 
        #may need to offset this by 1, since counting starts at 0




    def set_parent(self,cluster):
        #set the given cluster to be the parent of the cluster, i.e. first in stored data

        if not self.__has_parent:
            self.__stored_data.insert(0, self.__compute_coordinate(cluster))
            self.__has_parent = True

        return

    def kill(self, frame):
        # set this cluster to dead status

        self.__death_frame = frame
        self.__set_lifetime()
        self.__is_dead = True

        return

    def update_data(self, cluster, frame, monomer_frac):
        #append the current cluster's coordinate data to storage

        if (frame > self.__last_updated and not self.__is_dead):

            #compute a dictionary of requested values using this cluster
            self.__stored_data.append(self.__compute_coordinate(cluster))

            #append the monomer fraction when this cluster formed
            self.__stored_data[-1]['monomer_fraction'] = monomer_frac

            #updated the time of last update to current frame
            self.__last_updated = frame

        return

    def add_monomers(self, cluster, num_monomers, monomer_frac):
        #update the from_monomer list to denote a transition from monomer to cluster

        #compute a dictionary of requested values using this cluster
        self.__from_monomer.append((self.__compute_coordinate(cluster), num_monomers))

        #append the monomer fraction from previous timestep of cluster formation
        self.__from_monomer[-1][0]['monomer_fraction'] = monomer_frac

        return

    def remove_monomers(self, cluster, num_monomers):
        #update the to_monomer list to denote a transition from cluster to monomer

        self.__to_monomer.append((self.__compute_coordinate(cluster), num_monomers))
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


    def __set_lifetime(self):

        self.__lifetime = self.__death_frame - self.__birth_frame

        return

    def __compute_coordinate(self, cluster):
        '''this computes various observables for the cluster, based on user input
           given to the observer class. Default is simply number of bodies'''


           #move computation of properties to the observer class

        #init a dict to store properties of the cluster
        property_dict = dict()

        #get the list of observables requested, and compute them from cluster
        observables = self.__observer.get_observables()
        for obs in observables:

            if obs == "num_bodies":

                property_dict['num_bodies'] = len(cluster.get_bodies())

            elif obs == "positions":

                property_dict['positions'] = cluster.get_body_positions()

            else:

                raise("The requested property is not implemented. Check that it is"\
                    " implemented and spelled correctly")


            


        return property_dict





class Observer:

    def __init__(self, gsd_file):

        #use the gsd file to determine an output file for this run - todo
        self.__outfile = "test_out.py"

        #todo - determine obersvables fromm an input file


        #init a set of observables to compute. Always add monomer fraction
        self.__observable_set = set()

        #init variables for current and previous monomer fractions
        self.current_monomer  = -1
        self.previous_monomer = -1



    def add_observable(self, observable):

        self.__observable_set.add(observable)

    def get_observables(self):

        return self.__observable_set

    def init_test_set(self):
        #init the observable set to those helpful for testing

        self.__observable_set = set(['positions'])









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

def get_possible_matches(cluster, old_bodies):
    #determine which previous clusters each body in current cluster was part of

    #init a set to store possible cluster matches
    possibleMatches = set()

    #loop over bodies in cluster. get their cluster_id from old_bodies. add to set
    for body_id in cluster.get_body_ids():

        if len(old_bodies) > 0:
            old_body = old_bodies[body_id]
            old_cluster = old_body.get_cluster()
            # print(body_id, old_cluster)
            if old_cluster is not None:
                possibleMatches.add(old_cluster)

    return possibleMatches

def get_monomer_stats(old_ids, new_ids, old_bodies, new_bodies):
    #determine the number of monomer additions and subtractions from old to new cluster

    #init counters for number of monomers gained and lost
    monomers_gained = 0
    monomers_lost   = 0

    #first, do additions. look at ids in new but not old, and check cluster in old_bodies
    new_not_old = new_ids.difference(old_ids)
    for entry in new_not_old:

        cluster_id = old_bodies[entry].get_cluster_id()
        if cluster_id == -1:
            monomers_gained += 1

    #next do subtractions. look at ids in old but not new. check cluster in bodies
    old_not_new = old_ids.difference(new_ids)
    for entry in old_not_new:

        cluster_id = new_bodies[entry].get_cluster_id()
        if cluster_id == -1:
            monomers_lost += 1

    return monomers_gained, monomers_lost



def update_clusters(clusters, cluster_info, bodies, old_bodies, frame, observer):
    #use the existing clusters to assign an id and update cluster

    #make a queue for the new clusters and dict to store the mappings
    queue = [cluster for cluster in clusters] #TODO change to list.copy()
    label_dict = dict()

    #make a list to store death updates for merging - entries (dying id, new cluster id)
    merge_updates = []

    #make a dict to store tentative updates during the matching. Apply at the end
    tentative_updates = dict()

    #loop over the queue to assign labels to each new cluster
    while len(queue) > 0:

        #grab the first cluster in queue
        cluster = queue.pop(0)

        #determine which previous clusters are possible matches for the current cluster
        possibleMatches = get_possible_matches(cluster, old_bodies)

        if len(possibleMatches) == 0:

            '''
            If there are no possibleMatches, none of the bodies in this cluster were in clusters
            during the previous timestep. Thus this is a new cluster. Create a new cluster whose
            id is the number of elements in cluster_info.
            '''

            #get the new cluster id
            cluster_num = len(cluster_info)

            #update the cluster with the new id, add entry to cluster_info
            cluster.set_cluster_id(cluster_num)
            label_dict[cluster_num] = [0, cluster]
            cluster_info.append(ClusterInfo(cluster, frame, observer))

            num_bodies = len(cluster.get_bodies())
            monomer_frac = observer.previous_monomer
            cluster_info[cluster_num].add_monomers(cluster, num_bodies, monomer_frac)
            # print("created new cluster with id ", cluster_num)

        elif len(possibleMatches) == 1:

            '''
            If there is exactly one possible match, then all the bodies in this cluster were 
            either in the same cluster, or not part of a cluster during the previous timestep. 
            Several cases are possible.
            1) The structure is exactly the same. 
            2) The structure has gained monomers 
            3) The structure has broken into sub-structures
            '''

            #get the match and get its cluster id
            match = list(possibleMatches)[0]
            match_id = match.get_cluster_id()

            #perform a set difference to determine gain/loss of subunits
            old_not_new = set(match.get_body_ids()).difference(set(cluster.get_body_ids()))
            new_not_old = set(cluster.get_body_ids()).difference(set(match.get_body_ids()))

            # print(old_not_new, new_not_old)

            #define a similarity, max of the two set differences
            similarity = max(len(old_not_new), len(new_not_old))
            # similarity = len(old_not_new)
            # print("Similarity (Break): ", similarity)
            # print("set diffs: ", old_not_new, new_not_old)

            #if two or more particles break off
            if len(old_not_new) > 1:

                #check if this match has been assigned already
                if match_id in label_dict: #removed .keys() in case this breaks

                    #check if the current similarity score is better 
                    if similarity < label_dict[match_id][0]:

                        #TODO implement a steal id that pops the id and sets it to -1

                        #current cluster is more similar to match. overwrite old cluster
                        old_cluster = label_dict[match_id][1]
                        old_cluster.set_cluster_id(-1)

                        #set label for the new cluster
                        #match.update(cluster)
                        tentative_updates[match] = cluster
                        cluster.set_cluster_id(match_id)
                        label_dict[match_id] = [similarity, cluster]

                        # print("Better match found. Updated cluster match to ", match_id)

                        #add the old cluster to the queue for reassignment
                        queue.append(old_cluster)

                    #if not, assign a new cluster
                    else: 

                        #get the new cluster id
                        cluster_num = len(cluster_info)

                        #update the cluster with the new id, add entry to cluster_info
                        cluster.set_cluster_id(cluster_num)
                        label_dict[cluster_num] = [0, cluster]
                        cluster_info.append(ClusterInfo(cluster, frame, observer))
                        cluster_info[-1].set_parent(match)
                        # print("Match found with worse similarity. Created new cluster with id ", cluster_num)

                #this matching old cluster has not been assigned, so assign it
                else:

                    #update the old cluster with new. This sets the old_bodies references to cluster
                    #match.update(cluster)
                    tentative_updates[match] = cluster

                    #grab the cluster id and set it for the new cluster
                    cluster_id = match.get_cluster_id()
                    cluster.set_cluster_id(cluster_id)
                    label_dict[cluster_id] = [similarity, cluster]
                    
                    # print("matched cluster to existing ", cluster_id)


            #if the difference is 1 or 0, same cluster or only lost one monomer
            else:

                #update the old cluster with new. This sets the old_bodies references to cluster
                #match.update(cluster)
                tentative_updates[match] = cluster

                #grab the cluster id and set it for the new cluster
                cluster_id = match.get_cluster_id()
                cluster.set_cluster_id(cluster_id)
                label_dict[cluster_id] = [similarity, cluster]
                
                # print("matched cluster to existing ", cluster_id)

        else:

            '''
            If there is more than one potential match, that means that two or more clusters from the
            previous timestep have coalesced during this frame. If one cluster is bigger than the 
            others, keep that one. 
            May also be able to get here if cluster merge and break at the same time. 
            '''
            
            # print("Clusters have merged")

            #get a list of possible matches, and assign a similarity to each
            possible_list   = list(possibleMatches)
            similarity_vals = []
            for possible_match in possible_list:

                #do a set difference between cluster and possible
                old_not_new = set(possible_match.get_body_ids()).difference(set(cluster.get_body_ids()))
                new_not_old = set(cluster.get_body_ids()).difference(set(possible_match.get_body_ids()))
                similarity = max(len(old_not_new), len(new_not_old))
                # similarity = len(new_not_old)
                # print("Similarity (Merge): ", similarity)

                similarity_vals.append(similarity)

            #sort the possibilities by similarity
            poss_aug = [(similarity_vals[i], i) for i in range(len(similarity_vals))]
            poss_aug.sort()
            sorted_sim,permutation = zip(*poss_aug)
            sorted_poss = [possible_list[i] for i in permutation]

            #loop through sorted list. assign this cluster the first un-assigned match
            match_flag = False
            for possibility in sorted_poss:

                #check if match not already assigned. If not, assign it. 
                match_id = possibility.get_cluster_id()
                if (not match_flag):
                    if (match_id not in label_dict): #removed .keys() in case things break

                        # possibility.update(cluster)
                        tentative_updates[possibility] = cluster

                        #set id for the new cluster
                        cluster.set_cluster_id(match_id)
                        label_dict[match_id] = [similarity, cluster]
                        match_flag = True
                        assigned_id = match_id

                    else:
                        continue

                #match is already assigned, set other possibilities to update to it and die 
                else:

                    merge_updates.append((match_id, assigned_id))

            #if no match has been made after all possibilities, create new cluster
            if not match_flag:

                #i dont think its possible to get here, but we shall see
                # print("All clusters assigned. Creating new cluster")

                #get the new cluster id
                cluster_num = len(cluster_info)

                #update the cluster with the new id, add entry to cluster_info
                cluster.set_cluster_id(cluster_num)
                label_dict[cluster_num] = [0, cluster]
                cluster_info.append(ClusterInfo(cluster, frame, observer))
                # print("created new (merge) cluster with id ", cluster_num)


    #loop over the tentative updates and apply them. do monomer checking first
    for key in tentative_updates.keys():

        #get lists of the body ids in the pre and post update clusters
        past_ids = set(key.get_body_ids())
        new_ids  = set(tentative_updates[key].get_body_ids())

        #determine how many monomers were gained and lost from these id sets
        m_gain, m_lost = get_monomer_stats(past_ids, new_ids, old_bodies, bodies)
        # print("Monomer gain {}, Monomer loss {}".format(m_gain, m_lost))

        #update the monomer stats
        if m_gain > 0:

            cluster_id = key.get_cluster_id()
            monomer_frac = observer.previous_monomer
            cluster_info[cluster_id].add_monomers(tentative_updates[key], m_gain, monomer_frac)

        if m_lost > 0:

            cluster_id = key.get_cluster_id()
            cluster_info[cluster_id].remove_monomers(key, m_lost)


        #perform the update on the cluster
        key.update(tentative_updates[key])

    #loop over the now matched clusters and update the cluster_info
    #(can loop over all keys in the label_dict to avoid unnec work)
    for key in list(label_dict.keys()):

        #get the cluster
        updated_cluster = label_dict[key][1]

        #update the clusterinfo
        cluster_info[key].update_data(updated_cluster, frame, observer.current_monomer)
        # print("Updated cluster {}".format(key))

    #loop over the clusters that merged and are no longer tracked
    for update_pair in merge_updates:

        old_id = update_pair[0]
        current_id = update_pair[1]
        current_cluster = label_dict[current_id][1]

        #update to cluster, but also set a death
        cluster_info[old_id].update_data(current_cluster, frame, observer.current_monomer)
        cluster_info[old_id].kill(frame)
        # print("Updated and killed cluster {} (Merge)".format(old_id))


    return clusters, cluster_info




def track_clustering(snap, sim, frame, cluster_info, old_bodies, observer):
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
    num_monomers = 0
    for group in G:

        #check for dimers or larger, create a cluster object
        if len(group) > 1:

            #extract the involved bodies from the group and create a cluster
            body_list = [bodies[q] for q in group]
            clusters.append(Cluster(body_list, frame))

        #increment the number of free monomers
        else:
            num_monomers += 1

    #set the monomer fraction
    monomer_frac = num_monomers / len(bodies)
    print(monomer_frac)
    observer.current_monomer = monomer_frac

    #map the clusters from previous timestep to this step to assign labels
    print("Frame {} updates:".format(frame))
    clusters, cluster_info = update_clusters(clusters, cluster_info, bodies, old_bodies, 
                                             frame, observer)

    #update the previous monomer frac
    observer.previous_monomer = monomer_frac

    #return updated cluster info and the bodies to be used as old_bodies next frame
    return cluster_info, bodies