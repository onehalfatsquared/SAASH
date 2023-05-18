'''

Implementation of a Frame class. This class stores all relevant information from a single
frame of the hoomd simulation in order to perform cluster matching across frames. 

A Frame is constructed from the bodies and clusters that are present in each frame, as well
as the frame number and the fraction of subunits that are monomers during the frame. 

The main function in the class is the update() method, which takes in a list of ClusterInfo 
objects and the previous Frame in order to match the old clusters to clusters in the current 
Frame. This handles the creation of new clusters, as well as the merging and splitting of old 
clusters, and monomer gain/loss. 

'''

from . import body as body
from . import cluster as clust

import numpy as np
from collections import defaultdict

from enum import Enum

class Event(Enum):

    NEW_CLUSTER    = 0
    POSSIBLE_SPLIT = 1
    MERGE          = 2
    SPLIT          = 3
    PERSIST        = 4


class Frame:

    def __init__(self, bodies, clusters, frame_num, monomer_ids, monomer_frac = -1):

        #init a dictionary that stores which id gets mapped to which cluster
        self.__label_dict = dict()

        #init a dictionary that stores which old clusters update to new ones
        self.__updates    = dict()

        #make a list to store death updates for merging - entries (dying id, new cluster id)
        self.__merge_updates = []

        #init the clusters and bodies lists
        self.__clusters   = clusters
        self.__bodies     = bodies

        #init the other values
        self.__monomer_frac = monomer_frac
        self.__frame_num    = frame_num
        self.__monomer_ids  = set(monomer_ids)

        #set a tolerance for cluster absorbption fraction
        self.__absorb_tol = 0.49


    def create_first_frame(self, cluster_info, frame_num, observer):
        #for first frame, just create cluster info for all existing clusters

        for orig_cluster in self.__clusters:

            self.__add_new_cluster(cluster_info, orig_cluster, frame_num,  
                                   self.__monomer_frac, observer)

            #set the previous monomer concentration to be 1 for monomer -> cluster transition
            num_bodies = len(orig_cluster.get_bodies())
            cluster_info[-1].add_monomers(orig_cluster, frame_num, num_bodies, 1)

        return


    def update(self, cluster_info, old_frame, observer):
        #use the previous frame data to update the new frame and append to cluster_info

        #make a queue for matching the new clusters to the old ones
        queue = self.__clusters.copy()
        # queue = self.__clusters 

        #get required data from current and previous frame
        current_monomer = self.get_monomer_fraction()
        prev_monomer    = old_frame.get_monomer_fraction()
        bodies          = self.get_bodies()
        old_bodies      = old_frame.get_bodies()
        frame_num       = self.get_frame_num()

        #loop over the queue to assign labels to each new cluster
        while len(queue) > 0:

            #grab and remove the first cluster in queue
            cluster = queue.pop(0)

            ##print("\nLooking at cluster {}".format(cluster.get_body_ids()))

            #determine which previous clusters are possible matches for the current cluster
            possibleMatches = cluster.get_possible_matches(old_bodies)

            #get an event based on the possible matches
            event = self.__determine_event(possibleMatches)
            # ##print(event)

            if (event == Event.NEW_CLUSTER):
                #create a new cluster and append to cluster_info

                self.__add_new_cluster(cluster_info, cluster, frame_num,  
                                       current_monomer, observer)

                #augment with monomer addition info
                num_bodies = len(cluster.get_bodies())
                cluster_info[-1].add_monomers(cluster, frame_num, num_bodies, prev_monomer)

            elif (event == Event.POSSIBLE_SPLIT):
                #clusters have possibly split. perform matching if they have
               
                #extract the matching cluster from possibilities list
                match = list(possibleMatches)[0]

                #determine similarity between match and cluster. determine if split possible
                similarity, sub_event = get_similarity(match, cluster)
                # print(sub_event)

                if (sub_event == Event.SPLIT):
                    #if two or more particles split from the cluster. Find best match

                    to_queue = self.__handle_split(match, cluster, similarity, prev_monomer,
                                                   cluster_info, observer)
                    if (to_queue is not None):
                        queue.append(to_queue)


                #if the difference is 1 or 0, same cluster or only lost one monomer
                elif (sub_event == Event.PERSIST):

                    #print("Cluster Persist detected")
                    #print("Match is id {}, bodies {}".format(match.get_cluster_id(), match.get_body_ids()))
                    #check if this cluster is assigned already
                    if (match.get_cluster_id() in self.__label_dict):

                        #if the diff is 0 or 1, this is likely a better match, but check
                        #print("sim, old_sim = ", similarity, self.__label_dict[match.get_cluster_id()][0])
                        if similarity < self.__label_dict[match.get_cluster_id()][0]:

                            #print("Identity Stolen. Match is {}".format(match.get_body_ids()))
                            #steal the id from the matched cluster and add victim to queue
                            theft_victim = self.__assign_better_match(cluster, match, similarity)
                            queue.append(theft_victim)

                        #the match is worse, so make a new cluster
                        else:

                            self.__add_new_cluster(cluster_info, cluster, frame_num,  
                                                   current_monomer, observer)

                            #add to update list with a blank cluster object (for mon info update)
                            empty_cluster = clust.Cluster([], frame_num-1)
                            empty_cluster.set_cluster_id(cluster.get_cluster_id())
                            self.__updates[empty_cluster] = cluster


                    #otherwise, just assign the match 
                    else:
                        #print("matched cluster to existing ", match.get_cluster_id(), " - ", cluster.get_body_ids())
                        self.__assign_match(cluster, match, similarity)
                        
            elif (event == Event.MERGE):

                # #print("Clusters have merged")
                self.__handle_merge(possibleMatches, cluster, cluster_info, prev_monomer, observer)

        #apply all the tentatively computed updates
        self.__apply_updates(cluster_info, old_frame)

        return cluster_info

    #getter functions

    def get_frame_num(self):

        return self.__frame_num

    def get_bodies(self):

        return self.__bodies

    def get_clusters(self):

        return self.__clusters

    def get_monomer_fraction(self):

        return self.__monomer_frac

    def get_monomer_ids(self):

        return self.__monomer_ids

    def get_cluster_size_distribution(self, observer):
        #compute the distribution of cluster sizes in the frame

        #check if we are focusing on any particular cluster sizes
        focus_list = observer.get_focus_list()
        if focus_list is not None:

            #get cluster sizes and detailed info on sizes in the focus list
            size_dict, focus_dict = self.__get_cluster_sizes_focused(focus_list, observer)

        else:

            #just get the sizes
            size_dict = self.__get_cluster_sizes()

        #add the monomers that are not clustered to the dict
        size_dict[1] = len(self.__monomer_ids)

        #determine the largest group
        non_zero_counts = len(size_dict.values())
        if non_zero_counts > 0:
            largest_group_size = np.max(np.array(list(size_dict.keys())))
        else:
            largest_group_size = 0

        if focus_list is not None:
            return size_dict, largest_group_size, focus_dict
        else:
            return size_dict, largest_group_size


    def __get_cluster_sizes(self):
        #returns a dict where keys are sizes and values are number of clusters of that size

        '''init a dictionary to store histogram data
           number of clusters (value) of each size (key)'''
        size_dict = defaultdict(int)

        #loop over clusters and increment the corresponding size index
        for clust in self.__clusters:

            L = clust.get_num_bodies()
            size_dict[L] += 1

        return size_dict

    def __get_cluster_sizes_focused(self, focus_list, observer):
        #same as other get_sizes, but also tracks number of each microstate for sizes in list

        '''init a dictionary to store histogram data
           number of clusters (value) of each size (key)'''
        size_dict = defaultdict(int)

        #create a dict that stores time-series of number of each microstate
        # focus_dict = {cluster_size:[] for cluster_size in focus_list}
        focus_dict = {cluster_size:defaultdict(int) for cluster_size in focus_list}

        #loop over clusters and increment the corresponding size index
        for clust in self.__clusters:

            L = clust.get_num_bodies()
            size_dict[L] += 1

            if L in focus_list:
                #increment the counter for this particular microstate

                microstate_rep = self.__make_microstate_representation(clust, observer)
                focus_dict[L][microstate_rep] += 1

        return size_dict, focus_dict

    def __make_microstate_representation(self, cluster, observer):
        '''
        Constructs a representation of a microstate using all easily human-interpretable
        distinguishing properties set in the observer. 

        For example, imagine a linear polymer of two subunit types in the orientation ABBA.
        If "bonds" is set in the observer we will return (("A-B",2),("B-B",1)). 
        If "types" is set as well then it becomes (("A",2),("B",2),("A-B",2),("B-B",1))
        '''

        #get the list of observables to compute and grab them from cluster
        common = sorted(observer.get_identifying_observables(),reverse=True)
        values = [observer.compute_observable(cluster,obs) for obs in common]

        #sort each value for unique representations and flatten the list
        values = [sorted(quantities.items()) for quantities in values]
        values = sum(values, [])

        #return a tuple with the properties
        return tuple(values)


    #private utility functions for the update method

    def __determine_event(self, possibleMatches):
        #return what kind of update to perform based on length of possible matches

        L = len(possibleMatches)

        if L == 0:

            '''
            If there are no possibleMatches, none of the bodies in this cluster were 
            in clusters during the previous timestep. Thus this is a new cluster. 
            '''

            return Event.NEW_CLUSTER

        elif L == 1:

            '''
            If there is exactly one possible match, then all the bodies in this cluster were 
            either in the same cluster, or not part of a cluster during the previous timestep. 
            Several cases are possible.
            1) The structure is exactly the same. 
            2) The structure has gained monomers 
            3) The structure has broken into sub-structures
            '''

            return Event.POSSIBLE_SPLIT

        else:

            '''
            If there is more than one potential match, that means that two or more clusters 
            from the previous timestep have merged during this frame. 

            May also be able to get here if clusters merge and break at the same time. 
            '''

            return Event.MERGE


    def __add_new_cluster(self, cluster_info, cluster, frame_num, current_monomer, observer):
        #create new cluster and append it to cluster info

        #get the id number for the new cluster by looking at length of cluster info
        cluster_num = len(cluster_info)

        #update the cluster with the new id, add entry to cluster_info
        cluster.set_cluster_id(cluster_num)
        self.__label_dict[cluster_num] = [0, cluster]
        new_cluster = clust.ClusterInfo(cluster, frame_num, current_monomer, observer)
        cluster_info.append(new_cluster)

        #print("created new cluster with id ", cluster_num, ' - bodies ', cluster.get_body_ids())

        return

    def __assign_match(self, cluster, match, similarity):
        #assign cluster to its match in the previous frame. update relevant dict entries

        #grab the cluster id and set it for the new cluster
        cluster_id = match.get_cluster_id()
        cluster.set_cluster_id(cluster_id)

        #update the update and label dicts
        self.__updates[match] = cluster
        self.__label_dict[cluster_id] = [similarity, cluster]

        return

    def __assign_better_match(self, cluster, match, similarity):
        #cluster is closer to match than theft victim. steal ids and update dicts

        match_id = match.get_cluster_id()

        #current cluster is more similar to match. overwrite old cluster
        theft_victim = self.__label_dict[match_id][1]
        cluster.steal_id(theft_victim)

        #overwrite entries to the update and label dicts
        self.__updates[match] = cluster
        self.__label_dict[match_id] = [similarity, cluster]

        #return the theft victim
        return theft_victim


    def __handle_split(self, match, cluster, similarity, prev_monomer, cluster_info, observer):
        '''
        If a cluster splits, determine the best match as the one most similar to the
        original. It is possible for a match to be assigned, but a better one to be found 
        later, in which case we steal the ID from the previous match and return the old
        cluster to be added back to the queue. 
        '''

        #print("Cluster Split detected")

        #set frame num and monomer frac
        frame_num = self.get_frame_num()
        current_monomer = self.get_monomer_fraction()

        #get the id for the matched cluster
        match_id = match.get_cluster_id()

        #check if this match has been assigned already
        if match_id in self.__label_dict:

            #check if the current similarity score is better 
            if similarity < self.__label_dict[match_id][0]:

                theft_victim = self.__assign_better_match(cluster, match, similarity)
                #print("Better match found. Updated cluster match to ", match_id, "\n")

                #return the old cluster for reassignment to the queue
                return theft_victim

            #if similarity is worse, assign a new cluster
            else: 

                #since we start from parent info, frame num needs a -1 to be consistent
                self.__add_new_cluster(cluster_info, cluster, frame_num-1,  
                                       current_monomer, observer)

                #when splitting, set the parent to be the matched cluster
                cluster_info[-1].set_parent(match, prev_monomer)
                #print("Match found with worse similarity. Created new cluster with id ", cluster.get_cluster_id())

        #this matching old cluster has not been assigned, so assign it
        else:

            #if most of the bodies are missing, assign new cluster
            #get the ids for the match and assigned cluster
            cluster_ids = set(cluster.get_body_ids())
            matching_cluster_ids = set(match.get_body_ids())

            #print("Coming from match {} - {}".format(match.get_cluster_id(), match.get_body_ids()))

            #determine what fraction of subunits in matching are now in assigned 
            amount_in_both = len(cluster_ids.intersection(matching_cluster_ids))
            frac_in_common = amount_in_both / len(matching_cluster_ids)
            # #print(frac_in_common)

            #if a sufficient fraction is missing, make new cluster with it
            if frac_in_common < self.__absorb_tol:

                #since we start from parent info, frame num needs a -1 to be consistent
                self.__add_new_cluster(cluster_info, cluster, frame_num-1,  
                                       current_monomer, observer)

                #when splitting, set the parent to be the matched cluster
                cluster_info[-1].set_parent(match, prev_monomer)
                #print("Identified smaller part of a cluster split. Created new cluster with id ", cluster.get_cluster_id())
                #print("\n\n\n\n\n\n\n\n\n\nadfhsdfsbdjfsdf\n\n\n\n\n")

            #assign the match if it is a true split
            else:

                #print("Matched cluster to existing id {} - {}".format(match.get_cluster_id(), match.get_body_ids()))

                self.__assign_match(cluster, match, similarity)

                #print("Applied Split - {} split off from {}".format(cluster.get_body_ids(), match.get_body_ids()))
                #print("New id for {} is {}".format(cluster.get_body_ids(), cluster.get_cluster_id()))


        return None

    def __handle_merge(self, possibleMatches, cluster, cluster_info, prev_monomer, observer):
        '''
        If clusters merge together, determine which possible match is most similar to
        the new cluster and assign the id to the new cluster. The less similar clusters
        get absorbed to the new cluster and are set to be killed in the apply_update step. 
        '''

        #print("Merge detected.")

        #get a list of possible matches, and assign a similarity to each
        possible_list   = list(possibleMatches)
        similarity_vals = []

        for possible_match in possible_list:

            similarity, dummy_event = get_similarity(possible_match, cluster)
            similarity_vals.append(similarity)

        #sort the possibilities by similarity
        sorted_poss, sorted_similarity = sort_A_by_B(possible_list, similarity_vals)

        #check if first two of the similarities are equal
        if sorted_similarity[0] == sorted_similarity[1]:

            #further sort by cluster index
            c0 = sorted_poss[0].get_cluster_id()
            c1 = sorted_poss[1].get_cluster_id()

            #swap so the lower cluster index appears first
            if c1 < c0:
                sorted_poss[0], sorted_poss[1] = sorted_poss[1], sorted_poss[0]

        #loop through sorted list. assign this cluster the first un-assigned match
        match_flag = False
        for poss_index in range(len(sorted_poss)):
            
            possibility = sorted_poss[poss_index]
            similarity  = sorted_similarity[poss_index]

            #print("Possible match : ", possibility.get_body_ids(), similarity, possibility.get_cluster_id())
            #print("Match flag is {} currently".format(match_flag))
            #check if match not already assigned. If not, assign it. 
            match_id = possibility.get_cluster_id()
            if (not match_flag):
                if (match_id not in self.__label_dict):

                    self.__assign_match(cluster, possibility, similarity)
                    #print("Applied match to ", cluster.get_body_ids(), possibility.get_body_ids())
                    match_flag = True
                    assigned_id = match_id

            #match is already assigned, determine if other possibilities have been absorbed
            else:

                #get the ids for the possible clyster and assigned cluster
                cluster_ids = set(cluster.get_body_ids())
                matching_cluster_ids = set(possibility.get_body_ids())
                # #print(cluster_ids)
                # #print(matching_cluster_ids)

                #determine what fraction of subunits in matching are now in assigned 
                amount_in_both = len(cluster_ids.intersection(matching_cluster_ids))
                frac_in_common = amount_in_both / len(matching_cluster_ids)
                # #print(frac_in_common)

                #if a sufficient fraction is there, consider absorbed
                if frac_in_common > self.__absorb_tol:
                    self.__merge_updates.append((match_id, assigned_id))
                    #print("{} and {} set to merge".format(possibility.get_body_ids(), assigned_id))

        #if no match has been made after all possibilities, complicated edge case.
        if not match_flag:

            '''
            All the times I have arrived to this code block is when a resulting cluster
            contains broken off pieces of two or more previous clusters, and they have 
            already been assigned better matches. 

            I will construct a new cluster with the leftovers. May refine this at a later 
            data. TODO - think about refining this code block.
            '''

            #create a new cluster
            self.__add_new_cluster(cluster_info, cluster, self.get_frame_num(),  
                                   self.get_monomer_fraction(), observer)

            #augment with monomer addition info
            num_bodies = len(cluster.get_bodies())
            cluster_info[-1].add_monomers(cluster, self.get_frame_num(), 
                                          num_bodies, prev_monomer)

        return


    def __apply_updates(self, cluster_info, old_frame):
        '''
        All of the updates are currently stored in the dicts updates and label_dict. Here
        we apply the updates to the actual clusters, the cluster_info, and handle killing 
        of the merged clusters. 
        '''

        old_bodies = old_frame.get_bodies()
        bodies     = self.get_bodies()

        prev_monomer    = old_frame.get_monomer_fraction()
        current_monomer = self.get_monomer_fraction()

        frame_num = self.get_frame_num()

        #loop over the tentative updates and apply them. do monomer checking first
        for key in self.__updates.keys():

            #get lists of the body ids in the pre and post update clusters
            past_ids = set(key.get_body_ids())
            new_ids  = set(self.__updates[key].get_body_ids())

            #determine how many monomers were gained and lost from these id sets
            m_gain, m_lost = get_monomer_stats(past_ids, new_ids, old_bodies, bodies)
            # print(key.get_cluster_id())
            # print("Monomer gain {}, Monomer loss {}".format(m_gain, m_lost))

            #update the monomer stats
            if m_gain > 0:

                cluster_id = key.get_cluster_id()
                cluster_info[cluster_id].add_monomers(self.__updates[key], frame_num, m_gain, prev_monomer)

            if m_lost > 0:

                cluster_id = key.get_cluster_id()
                cluster_info[cluster_id].remove_monomers(key, frame_num, m_lost, prev_monomer)

            # print(key.get_cluster_id(), cluster_info[key.get_cluster_id()].get_data())
            # print(key.get_cluster_id(), cluster_info[key.get_cluster_id()].get_monomer_gain_data())
        
        #loop over the now matched clusters and update the cluster_info
        #(can loop over all keys in the label_dict to avoid unnec work)
        for key in list(self.__label_dict.keys()):

            #get the cluster
            updated_cluster = self.__label_dict[key][1]

            #update the clusterinfo
            cluster_info[key].update_data(updated_cluster, frame_num, current_monomer)

        #loop over the clusters that merged and are no longer tracked
        for update_pair in self.__merge_updates:

            old_id = update_pair[0]
            current_id = update_pair[1]
            current_cluster = self.__label_dict[current_id][1]

            #update to cluster, and set absorbed status
            cluster_info[old_id].update_data(current_cluster, frame_num, current_monomer)
            cluster_info[old_id].absorb(frame_num)

        #finally, check for structures that dissociate into monomers, not found by previous steps
        old_cluster_ids = set([c.get_cluster_id() for c in old_frame.get_clusters()])
        new_cluster_ids = set([c.get_cluster_id() for c in self.get_clusters()])
        old_not_new     = old_cluster_ids.difference(new_cluster_ids)

        for c_id in old_not_new:

            #if cluster is not dead or absorbed, that means it split to monomers. perform kill
            if not (cluster_info[c_id].is_dead() or cluster_info[c_id].is_absorbed()):
                cluster_match = None
                for poss_cluster in old_frame.get_clusters():
                    if c_id == poss_cluster.get_cluster_id():
                        cluster_match = poss_cluster
                        break

                #remove monomer and kill the cluster
                cluster_info[c_id].remove_monomers(cluster_match, frame_num, cluster_match.get_num_bodies(), prev_monomer)
                cluster_info[c_id].kill(frame_num)


        #print("\nIteration {} complete. The current ids in the label dict are:".format(frame_num))
        #print(list(self.__label_dict.keys()),"\n")

        return


####################################################################
############ Utitilty Functions for the Frame Update ###############
####################################################################


def get_similarity(old_cluster, new_cluster):
    #determine a similarity measure between old and new clusters
    #also return an event for split or persist

    #perform a set difference to determine gain/loss of subunits
    old_not_new = old_cluster.get_difference(new_cluster)
    new_not_old = new_cluster.get_difference(old_cluster)

    #define a similarity, max of the two set differences
    similarity = max(len(old_not_new), len(new_not_old))

    # #print(old_not_new)
    # #print(new_not_old)
    # #print(similarity)

    #determine if the cluster has split into a potentially new cluster or not
    event = Event.PERSIST
    if len(old_not_new) > 1:    
        event = Event.SPLIT

    return similarity, event


def sort_A_by_B(A, B):
    #sort the entries of A by the values in B. B is assumed to take numeric values, A does not 

    #augment B with integer indices and sort it by the first value
    augmented_B = [(B[i],i) for i in range(len(B))]
    augmented_B.sort()

    #extract the permutation of the augmented indices in the sorted list and apply it to A
    sorted_B, permutation = zip(*augmented_B)
    sorted_A = [A[i] for i in permutation]

    return sorted_A, sorted_B


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


####################################################################
############ Frame Creation from HOOMD Data ########################
####################################################################

def get_data_from_snap(snap, sim, frame_num):
    #go from a snapshot to a frame object with bodies, clusters, and other data

    #get a list of bodies to analyze
    bodies = body.create_bodies(snap, sim)

    #init a dictionary to store the bonds present - init with empty lists for each body_id
    bond_dict = dict()
    for bod in bodies:
        bond_dict[bod.get_id()] = []

    #determine the bond network using the list of bodies
    body.get_bonded_bodies(bodies, sim, bond_dict)

    #determine groups of bonded structures
    G = clust.get_groups(bond_dict)

    #for each group, create a cluster
    clusters     = []
    monomer_ids  = []
    num_monomers = 0
    for group in G:

        #check for dimers or larger, create a cluster object
        if len(group) > 1:

            #extract the involved bodies from the group and create a cluster
            body_list = [bodies[q] for q in group]
            clusters.append(clust.Cluster(body_list, frame_num))
            # print(clusters[-1].get_bond_types())

        #increment the number of free monomers
        else:
            monomer_ids.append(group[0])
            num_monomers += 1


    #set the monomer fraction
    monomer_frac = num_monomers / len(bodies)
    
    #create a Frame object for this frame and return it
    current_frame = Frame(bodies, clusters, frame_num, monomer_ids, monomer_frac) 
    return current_frame