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

from body import body as body
from body import cluster as clust

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
        self.__monomer_ids  = monomer_ids


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

            #determine which previous clusters are possible matches for the current cluster
            possibleMatches = cluster.get_possible_matches(old_bodies)

            #get an event based on the possible matches
            event = self.__determine_event(possibleMatches)
            # print(event)

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

                    to_queue = self.__handle_split(match, cluster, similarity, 
                                                   cluster_info, observer)
                    if (to_queue is not None):
                        queue.append(to_queue)


                #if the difference is 1 or 0, same cluster or only lost one monomer
                elif (sub_event == Event.PERSIST):

                    #assign the match to cluster
                    self.__assign_match(cluster, match, similarity)
                    # print("matched cluster to existing ", cluster_id)

            elif (event == Event.MERGE):

                # print("Clusters have merged")
                self.__handle_merge(possibleMatches, cluster, cluster_info, observer)

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

        # print("created new cluster with id ", cluster_num)

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


    def __handle_split(self, match, cluster, similarity, cluster_info, observer):
        '''
        If a cluster splits, determine the best match as the one most similar to the
        original. It is possible for a match to be assigned, but a better one to be found 
        later, in which case we steal the ID from the previous match and return the old
        cluster to be added back to the queue. 
        '''

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
                # print("Better match found. Updated cluster match to ", match_id)

                #return the old cluster for reassignment to the queue
                return theft_victim

            #if similarity is worse, assign a new cluster
            else: 

                self.__add_new_cluster(cluster_info, cluster, frame_num,  
                                       self.__monomer_frac, observer)

                #when splitting, set the parent to be the matched cluster
                cluster_info[-1].set_parent(match)
                # print("Match found with worse similarity. Created new cluster with id ", cluster_num)

        #this matching old cluster has not been assigned, so assign it
        else:

            self.__assign_match(cluster, match, similarity)
            # print("matched cluster to existing ", cluster_id)


        return None

    def __handle_merge(self, possibleMatches, cluster, cluster_info, observer):
        '''
        If clusters merge together, determine which possible match is most similar to
        the new cluster and assign the id to the new cluster. The less similar clusters
        get absorbed to the new cluster and are set to be killed in the apply_update step. 
        '''

        #get a list of possible matches, and assign a similarity to each
        possible_list   = list(possibleMatches)
        similarity_vals = []

        for possible_match in possible_list:

            similarity, dummy_event = get_similarity(possible_match, cluster)
            similarity_vals.append(similarity)

        #sort the possibilities by similarity
        sorted_poss, sorted_similarity = sort_A_by_B(possible_list, similarity_vals)

        #loop through sorted list. assign this cluster the first un-assigned match
        match_flag = False
        for poss_index in range(len(sorted_poss)):
            
            possibility = sorted_poss[poss_index]
            similarity  = sorted_similarity[poss_index]

            #check if match not already assigned. If not, assign it. 
            match_id = possibility.get_cluster_id()
            if (not match_flag):
                if (match_id not in self.__label_dict):

                    self.__assign_match(cluster, possibility, similarity)
                    match_flag = True
                    assigned_id = match_id

            #match is already assigned, set other possibilities to update to it and die 
            else:

                self.__merge_updates.append((match_id, assigned_id))

        #if no match has been made after all possibilities, something went wrong?
        if not match_flag:

            #i dont think its possible to get here, raise error if we do
            raise("Unable to find a match for a merged cluster, which "\
                  "should be impossible. Exiting...")

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
            # print("Monomer gain {}, Monomer loss {}".format(m_gain, m_lost))

            #update the monomer stats
            if m_gain > 0:

                cluster_id = key.get_cluster_id()
                cluster_info[cluster_id].add_monomers(self.__updates[key], frame_num, m_gain, prev_monomer)

            if m_lost > 0:

                cluster_id = key.get_cluster_id()
                cluster_info[cluster_id].remove_monomers(key, frame_num, m_lost, prev_monomer)


            #perform the update on the cluster
            key.update(self.__updates[key])

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

            #update to cluster, but also set a death
            cluster_info[old_id].update_data(current_cluster, frame_num, current_monomer)
            cluster_info[old_id].kill(frame_num)

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

        #increment the number of free monomers
        else:
            monomer_ids.append(group[0])
            num_monomers += 1

    #set the monomer fraction
    monomer_frac = num_monomers / len(bodies)
    
    #create a Frame object for this frame and return it
    current_frame = Frame(bodies, clusters, frame_num, monomer_ids, monomer_frac) 
    return current_frame