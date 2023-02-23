'''

This file contains implementations of a State class, which stores some discretized set of 
information to represent a molecular configuration. 

Also includes classes to store a collection of unique states that have been visited 
in each simulation. These can be combined to form a global database of all states 
visited among all parameter sets and conditions, for use in MSM construction. 

Also contains references to the location in which the full state information can be 
found, for use in visualization or targetted sampling. 
'''

import sys, os
import pickle

import numpy as np

import hashlib
import json


class StateScraper:
    '''
    StateScraper is responsible for constructing the collection of all observed states
    from a trajectory file. Acts as an interface between an analysis script and 
    the StateCollection class.
    '''

    def __init__(self, folder):

        self.__folder = folder
        self.__collection = StateCollection()

        self.__pkl_file = ""
        self.__complete = False

        return

    def run_collection(self):
        #collect all states from the trajectory data

        #unpickle the data
        self.__pkl_file = self.__get_path_to_pickle()
        with open(self.__pkl_file, 'rb') as f:
            sim_results = pickle.load(f)

        #extract the info from the pickle
        cluster_info = sim_results.cluster_info

        #extract states from the cluster transition data
        self.__get_states_from_data(cluster_info)
        self.__complete = True

        return

    def save(self):
        #pickle the collection for later access

        #check if the collection has been performed
        if not self.__complete:
            print("Warning: Run the data collection before saving")
            return

        #collection is complete. set path to save results and call save
        out_file_path = self.__pkl_file.split(".pkl")[0]
        out_file_path += ".sref"
        self.__collection.save(out_file_path)

        return


    def __get_states_from_data(self, cluster_info):
        '''
        Calls the get_data() method on each trajectory in cluster info
        '''

        #loop over each cluster trajectory
        for i in range(len(cluster_info)):

            #get the current trajectory and loop over every 2nd entry 
            traj      = cluster_info[i]
            traj_data = traj.get_data()
            traj_len  = len(traj_data)
            for j in range(0, traj_len):

                #construct a state from each data point along the traj. check if new
                data  = traj_data[j]
                state = traj.construct_state(data) 
                if self.__collection.is_new(state):

                    #create a StateRep and add it to the collection
                    file      = self.__pkl_file
                    frame_num = traj.get_birth_frame() + j
                    indices   = data['indices']
                    state_rep = StateRef(file, frame_num, indices)
                    self.__collection.add_state(state_rep)

        return

    def __get_path_to_pickle(self):

        #find the pickled transition file
        for file in os.listdir(self.__folder):
            if file.endswith(".pkl"):
                return os.path.join(self.__folder, file)

        #return an error message that a 'pkl' file was not found'
        msg = "No .pkl trajectory file was found in {}".format(self.__folder)
        raise FileNotFoundError(msg)

        return


class StateCollection:

    '''
    StateCollection simply stores a dictionary where the keys are State objects
    and the values are the associated StateRef structs. 

    Keeps track of unique states found in a particular trajectory. Allows searching
    by State() keys to get a reference to where the full state info can be found. 
    Addition is overloaded to combine the unique states from two collections. 

    Can be loaded from a pickled class object.
    ''' 

    def __init__(self, load_folder = None):

        self.__state_refs = dict()

        self.__load_path  = ""
        self.__was_loaded = False

        if load_folder is not None:
            load(load_folder)
            self.__was_loaded = True

        return

    def save(self, save_loc):
        #save self to the specified location

        #check if the collection was loaded, and save/load paths are same
        if (self.__was_loaded and self.__load_path == save_file):

            err_msg = "This save would overwrite the original file at {}".format(self.__was_loaded)
            raise RuntimeError(err_msg)

        #pickle self to the outfile location
        with open(save_loc,'wb') as outfile:
            pickle.dump(self, outfile)

        return

    def load(self, load_folder):
        #load the StateCollection object from the specified folder

        #search the folder for .sref extensions
        file_found = False
        for file in os.listdir(load_folder):
            if file.endswith(".sref"):
                self.__load_path = os.path.join(load_folder, file)
                file_found = True
                break

        if not file_found:

            #return an error message that a 'sref' file was not found
            msg = "No .sref file was found in {}".format(self.__folder)
            raise FileNotFoundError(msg)

        #unpickle the file and set the dicts equal
        with open(self.__load_path, 'rb') as f:
            saved_collection = pickle.load(f)

        self.__state_refs = saved_collection.get_dict().copy()

        return

    def add_state(self, state, state_ref):

        self.__state_refs[state] = state_ref

        return

    def is_new(self, state):
        #determine if state is not present in the dict already

        if state in self.__state_refs:
            return False

        return True

    def get_ref(self, state):

        if state in self.__state_refs:
            return self.__state_refs[state]

        msg = "{} not found in collection.".format(state)
        raise KeyError(msg)

        return

    def get_dict(self):

        return self.__state_refs


    def __add__(self, other_collection):
        #add in all states present in other_collection not in self

        for key,value in other_collection.get_dict().items():

            if key not in self.__state_refs:
                self.__state_refs[key] = value

        return


class StateRef:

    '''
    StateRef includes all the relevant information to identify the full information 
    for the associated state. These are:
        1) Location of the gsd file it came from
        2) The frame number the state was observed
        3) The indices of the bodies making up the state

    Ex: State(5,{bonds}) can be found in file path/ex.gsd on frame 712, consisting of bodies
    [4,9,123,432,506]. The full configuration of positions and orientations can then be found
    using this data.
    '''

    def __init__(self, file, frame_num, indices):

        self.__file_loc  = file
        self.__frame_num = frame_num
        self.__indices   = indices

        return

    def get_file(self):

        return self.__file_loc

    def get_frame(self):

        return self.__frame_num

    def get_indices(self):

        return self.__indices


class State:

    '''
    State will store all necessary information to represent a discretized form
    of a particular molecular configuration. The 'size' property will always be used to
    describe a state. Any other propoerties that depend on the system/situation will 
    be stored in the properties dict. 

    States will need to be stored and searchable in a database, so they will need to 
    be hashable. A hashing strategy based on immutifying the properties dict is used. 
    '''

    def __init__(self, size, properties = dict()):

        self.__size = size
        self.__properties = properties

        return


    def get_size(self):

        return self.__size
        
    def get_properties(self):
        
        return self.__properties

    def get_hash(self):
            
        dict_hash = self.__hash_dictionary(self.__properties)
        return hash(self.__size) ^ dict_hash

        
    def __hash__(self):
        
        return self.get_hash()
        
    def __eq__(self, state2):
        
        if self.__size != state2.get_size():
            return False

        if self.__properties != state2.get_properties():
            return False
            
        return True

    def __str__(self):

        all_properties = set(self.__properties.keys())

        if len(all_properties) == 0:

            return "State({},None)".format(self.get_size())

        else:

            state_str = "State({},{}) ".format(self.get_size(), all_properties)
            state_str+= "object with properties: {}".format(self.get_properties())
            return state_str

    def __repr__(self):
        
        return self.__str__()

    #dictionary hashing methods taken from:
    # https://ardunn.us/posts/immutify_dictionary/

    def __immutify_dictionary(self, d):
    #return an immutable copy of the provided dict, for use in hashing

        d_new = {}

        for k, v in d.items():
        
            # convert to python native immutables
            if isinstance(v, np.ndarray):
                d_new[k] = tuple(v.tolist())

            # immutify any lists
            elif isinstance(v, list):
                d_new[k] = tuple(v)

            # recursion if nested
            elif isinstance(v, dict):
                d_new[k] = self.__immutify_dictionary(v)

            # ensure numpy "primitives" are casted to json-friendly python natives
            else:
                # convert numpy types to native
                if hasattr(v, "dtype"):
                    d_new[k] = v.item()
                else:
                    d_new[k] = v
        
        return dict(sorted(d_new.items(), key=lambda item: item[0]))

    def __hash_dictionary(self, d):
        # Make a json string from the sorted dictionary
        # then hash that string

        d_hashable = self.__immutify_dictionary(d)
        s_hashable = json.dumps(d_hashable).encode("utf-8")
        m = hashlib.sha256(s_hashable).hexdigest()

        return hash(m)
