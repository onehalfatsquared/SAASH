'''

This file contains an implementation of a State class, which stores some discretized set of 
information to represent a molecular configuration. 

There are cases in which one would like to go from the discretized State description, back 
to a full description of the state (visualization or initialization for targetted sampling).

The StateScraper class will go through a cluster-processed trajectory file and find all 
unique states from that simulation. It creates a StateRef, the minimal info required to 
find out the full state information (gsd file, frame number, and body indices), for each
state and collects it in a StateRefCollection object.

The StateRefCollection can then be used to build a StateRepCollection, a mapping from a 
State object to a StateRep. This contains all the information needed to initialize the 
full molecular configuration (body positions, body type ids, and body orientations). 

Alternatively, a StateFrameCollection can be built, which recreates the full simulation 
snapshot that a particular state came from. 


'''

import sys, os
import pickle

import gsd.hoomd

import numpy as np

import hashlib
import json

import warnings

class StateScraper:
    '''
    StateScraper is responsible for constructing the collection of all observed states
    from a trajectory file. Acts as an interface between an analysis script and 
    the StateRefCollection class.
    '''

    def __init__(self, folder, verbose = False):

        self.__folder = folder
        self.__collection = StateRefCollection()

        self.__cl_file  = ""
        self.__verbosity = verbose

        self.__run_collection()
        self.__save()

        return

    def __run_collection(self):
        #collect all states from the trajectory data

        #unpickle the cluster data
        self.__cl_file = self.__get_path_to_cluster_data()
        with open(self.__cl_file, 'rb') as f:
            sim_results = pickle.load(f)

        #extract the info from the pickle
        cluster_info = sim_results.cluster_info

        #extract states from the cluster transition data
        print("Extracting states from {}".format(self.__cl_file))
        self.__get_states_from_data(cluster_info)

        return

    def __save(self):
        #pickle the collection for later access

        #collection is complete. set path to save results and call save
        out_file_path = self.__cl_file.split(".cl")[0]
        out_file_path += ".sref"
        self.__collection.save(out_file_path)

        return


    def __get_states_from_data(self, cluster_info):
        '''
        Calls the get_data() method on each trajectory in cluster info
        '''

        #TODO: do this as function of monomer fraction?

        #loop over each cluster trajectory
        for i in range(len(cluster_info)):

            #get the current trajectory and loop over every 2nd entry 
            traj      = cluster_info[i]
            jump      = traj.get_frame_jump()
            traj_data = traj.get_data()
            traj_len  = len(traj_data)
            for j in range(0, traj_len):

                #construct a state from each data point along the traj. check if new
                data  = traj_data[j]
                state = traj.construct_state(data) 
                if self.__collection.is_new(state):

                    #create a StateRep and add it to the collection
                    file      = self.__cl_file.split(".cl")[0] + ".gsd"
                    frame_num = (traj.get_birth_frame() + j) * jump
                    indices   = data['indices']
                    state_rep = StateRef(file, frame_num, indices)
                    self.__collection.add_state(state, state_rep)

                    if (self.__verbosity):
                        ns = self.__collection.get_num_states()
                        print("{}, Discovered {}".format(ns, state))

        print("Found {} states".format(self.__collection.get_num_states()))

        return

    def __get_path_to_cluster_data(self):

        #find the pickled transition file
        for file in os.listdir(self.__folder):
            if file.endswith(".cl"):
                return os.path.join(self.__folder, file)

        #return an error message that a 'cl' file was not found'
        msg = "No .cl trajectory file was found in {}".format(self.__folder)
        raise FileNotFoundError(msg)

        return


class StateRefCollection:

    '''
    StateRefCollection simply stores a dictionary where the keys are State objects
    and the values are the associated StateRef structs. 

    Keeps track of unique states found in a particular trajectory. Allows searching
    by State() keys to get a reference to where the full state info can be found. 
    Addition is overloaded to combine the unique states from two collections. 

    Can be loaded from a pickled class object.
    ''' 

    def __init__(self, load_location = None):

        self.__state_refs = dict()

        self.__load_path  = ""
        self.__was_loaded = False

        if load_location is not None:
            self.load(load_location)
            self.__was_loaded = True

        return

    def save(self, save_loc):
        #save self to the specified location

        #check if the collection was loaded, and save/load paths are same
        if (self.__was_loaded and self.__load_path == save_loc):

            err_msg = "This save would overwrite the original file at {}".format(self.__load_path)
            raise RuntimeError(err_msg)

        #check if save_loc is a directory, if so, give default name
        if os.path.isdir(save_loc):
            if not save_loc.endswith('/'):
                save_loc += "/"
            save_loc += 'state_database.sref'

        #if the save_loc doesnt end in sref, append the extension
        if not save_loc.endswith('.sref'):
            save_loc += '.sref'

        #pickle self to the outfile location
        with open(save_loc,'wb') as outfile:
            pickle.dump(self, outfile)

        return

    def load(self, load_location):
        #load the StateRefCollection object from the specified folder/file

        file_found = False

        #determine if a folder or a 'sref' file was provided
        if load_location.endswith(".sref"):

            self.__load_path = load_location
            file_found = True

        elif os.path.isdir(load_location):

            #search the folder for .sref extensions
            for file in os.listdir(load_location):
                if file.endswith(".sref"):

                    #check if there are several potential files and give error
                    if (file_found):
                        msg = "Multiple .sref files found in {}. Specify a single file".format(self.__folder)
                        raise RuntimeError(msg)

                    #set path to file loc and flag that a file was found
                    self.__load_path = os.path.join(load_location, file)
                    file_found = True

        else:

            self.__fnf_error(load_location)
       

        if not file_found:

            self.__fnf_error(load_location)

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

        return self.__state_refs.copy()

    def get_num_states(self):

        return len(self.__state_refs)

    def get_load_path(self):

        return self.__load_path


    def __add__(self, other_collection):
        #add in all states present in other_collection not in self

        for key,value in other_collection.get_dict().items():

            if key not in self.__state_refs:
                self.__state_refs[key] = value

        return self

    def __fnf_error(self, location):
        #raise a file not found error at the current location

        msg = "Provided load location ({}) could not be found. ".format(location)
        msg+= "Please provide a .sref file, or folder containing exactly one .sref file"
        raise FileNotFoundError(msg)
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


class StateRepCollection:

    '''
    StateRepCollection stores a dictionary where the keys are State objects
    and the values are the associated StateRep structs. 

    Is initialized with a StateRefCollection, containing references to the location
    of all the unique states encountered in a simulation. The class then follows these
    references to grab the required information to construct a StateRep.

    Can also be init by providing a folder containing a single .srep file, or a 
    path to a .srep file. 

    '''

    def __init__(self, initMethod = None):

        #check which init method is requested - StateRefCollection or string with path
        if isinstance(initMethod, StateRefCollection):

            method = "construct"
            refCollection = initMethod

        elif isinstance(initMethod, str):

            method = "load"
            load_location = initMethod

        elif initMethod is None:

            method = "none"

        else:

            err_msg = "To init StateRepCollection, please provide either a StateRefCollection\
                       or path to a folder containing a .srep file to load from."
            raise TypeError(err_msg)


        #init the class variables
        self.__state_reps = dict()

        self.__save_path  = ""
        self.__load_path  = ""
        self.__was_loaded = False

        #branch based on init method
        if method == "construct":

            self.__construct_rep_collection(refCollection)
            self.__set_save_path(refCollection)
            self.save(self.__save_path)

        elif method == "load":

            self.load(load_location)
            self.__was_loaded = True

        return

    def get_dict(self):

        return self.__state_reps.copy()

    def get_num_states(self):

        return len(self.__state_reps)

    def get_rep(self, state):

        if state in self.__state_reps:
            return self.__state_reps[state]

        msg = "{} not found in collection.".format(state)
        raise KeyError(msg)

        return

    def save(self, save_loc):
        '''
        Save self to the internal save path. User has no access to this function b/c
        this class is intended to be constructed once and not editied after.
        The save location is the same path as the refCollection the class is init with,
        but with .srep file extension
        '''

        #check if the collection was loaded, and save/load paths are same
        if (self.__was_loaded and self.__load_path == save_loc):

            err_msg = "This save would overwrite the original file at {}".format(self.__load_path)
            raise RuntimeError(err_msg)

        #check if save_loc is a directory, if so, give default name
        if os.path.isdir(save_loc):
            if not save_loc.endswith('/'):
                save_loc += "/"
            save_loc += 'state_database.srep'

        #if the save_loc doesnt end in sref, append the extension
        if not save_loc.endswith('.srep'):
            save_loc += '.srep'

        #pickle self to the outfile location
        with open(save_loc,'wb') as outfile:
            pickle.dump(self, outfile)

        return

    def load(self, load_location):
        #load the StateRefCollection object from the specified folder

        file_found = False

        #determine if a folder or a 'sref' file was provided
        if load_location.endswith(".srep"):

            self.__load_path = load_location
            file_found = True

        elif os.path.isdir(load_location):

            #search the folder for .sref extensions
            for file in os.listdir(load_location):
                if file.endswith(".srep"):

                    #check if there are several potential files and give error
                    if (file_found):
                        msg = "Multiple .srep files found in {}. Specify a single file".format(self.__folder)
                        raise RuntimeError(msg)

                    #set path to file loc and flag that a file was found
                    self.__load_path = os.path.join(load_location, file)
                    file_found = True

        else:

            self.__fnf_error(load_location)
       

        if not file_found:

            self.__fnf_error(load_location)

        #unpickle the file and set the dicts equal
        with open(self.__load_path, 'rb') as f:
            saved_collection = pickle.load(f)

        self.__state_reps = saved_collection.get_dict().copy()

        return


    def __construct_rep_collection(self, refCollection):

        #get the dict of refs from the collection
        refDict = refCollection.get_dict()

        #loop over all states (keys), create a StateRep using the Ref to find the data
        for state in refDict.keys():

            stateRef = refDict[state]
            stateRep = self.__create_rep(stateRef)
            self.__state_reps[state] = stateRep

        return



    def __create_rep(self, stateRef):
        #use the data in stateRef to get the necessary info to create a stateRep

        #get the identifying data
        file    = stateRef.get_file()
        frame   = stateRef.get_frame()
        indices = stateRef.get_indices()

        #open the file and frame referenced
        snaps = gsd.hoomd.open(name=file, mode="rb")
        L = len(snaps)
        if frame == L:
            frame -= 1
        snap  = snaps[frame]

        #filter the bodies according to index and grab the required data
        positions    = snap.particles.position[indices]
        types        = snap.particles.typeid[indices]
        typeids      = [snap.particles.types[t] for t in types]
        orientations = snap.particles.orientation[indices]

        #shift positions such that the CoM is the origin
        box       = snap.configuration.box
        box       = box[np.where(box > 1e-6)] #get rid of the skew components
        positions = self.__shift_to_origin(positions, box)

        #close file
        snaps.close()

        #create and return a StateRep object 
        return StateRep(len(types), positions, typeids, orientations)


    def __shift_to_origin(self, positions, box):
        #shift all positions such that the center of mass is the origin
        #take into account periodic boundary conditions

        #we start by moving all particles to the same quadrant/octant
        self.__shift_to_same_region(positions, box)

        #compute the center of mass
        CoM = self.__compute_center(positions.copy())

        #subtract CoM from all particles to center at the origin
        for i in range(len(positions)):
            positions[i] -= CoM

        return positions
        

    def __shift_to_same_region(self, positions, box):

        #start by choosing reference particle (arbitrary, use 0). compute displacements
        reference_pos = positions[0]
        displacements = positions - reference_pos

        #decompose displacements into a magnitude and a sign
        magnitudes    = np.abs(displacements)
        #surpress divide by 0 warning, will never be accessed since 0 < box_L
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            signs     = np.divide(magnitudes, displacements)

        #loop over each particle
        for i in range(len(positions)):

            #loop over each dimension
            for dim in range(len(box)):

                #check for all displacements larger than half the box size, add L
                if magnitudes[i][dim] > box[dim]/2:
                    positions[i][dim] -= signs[i][dim] * box[dim]

        return

    def __compute_center(self, positions):
        #get the center of mass of a collection of points (assume fixed mass)

        L = len(positions)
        CoM = positions[0]

        for i in range(1,L):
            CoM += positions[i]

        return (CoM / L)

    def __set_save_path(self, refCollection):
        #save to the same folder as refCollection was in

        ref_path = refCollection.get_load_path()
        self.__save_path = ref_path.split(".sref")[0] + ".srep"

        return


    def __add__(self, other_collection):
        #add in all states present in other_collection not in self

        for key,value in other_collection.get_dict().items():

            if key not in self.__state_reps:
                self.__state_reps[key] = value

        return self

    def __fnf_error(self, location):
        #raise a file not found error at the current location

        msg = "Provided load location ({}) could not be found. ".format(location)
        msg+= "Please provide a .srep file, or folder containing exactly one .srep file"
        raise FileNotFoundError(msg)
        return


class StateRep:

    '''
    StateRep ontains the minimum set of information required to represent a 
    full configuration of a particular state. These are:
        1) The number of bodies making up the state
        2) The positions of each body
        3) The type of each body
        4) The orientation of each body
        5) Others???
    '''

    def __init__(self, size, positions, types, orientations):  

        self.__size         = size
        self.__positions    = positions
        self.__types        = types
        self.__orientations = orientations


    def get_size(self):

        return self.__size

    def get_positions(self):

        return self.__positions

    def get_types(self):

        return self.__types

    def get_orientations(self):

        return self.__orientations


class StateFrameCollection:

    '''
    StateRepCollection stores a dictionary where the keys are State objects
    and the values are HOOMD gsd snapshots containing that state. 

    Is initialized with a StateRefCollection, containing references to the location
    of all the unique states encountered in a simulation. The class then opens
    the file from the Ref, loads the given frame, and saves it in the dict.

    Can also be init by providing a folder containing a single .sframe file, or a 
    path to a .sframe file, which is then loaded. 

    '''

    def __init__(self, initMethod = None):

        #check which init method is requested - StateRefCollection or string with path
        if isinstance(initMethod, StateRefCollection):

            method = "construct"
            refCollection = initMethod

        elif isinstance(initMethod, str):

            method = "load"
            load_location = initMethod

        elif initMethod is None:

            method = "none"

        else:

            err_msg = "To init StateFrameCollection, please provide either a StateRefCollection\
                       or path to a folder containing a .sframe file to load from."
            raise TypeError(err_msg)


        #init the class variables
        self.__state_frames = dict()

        self.__save_path  = ""
        self.__load_path  = ""
        self.__was_loaded = False

        #branch based on init method
        if method == "construct":

            self.__construct_frame_collection(refCollection)
            self.__set_save_path(refCollection)
            self.save(self.__save_path)

        elif method == "load":

            self.load(load_location)
            self.__was_loaded = True

        return

    def get_dict(self):

        return self.__state_frames.copy()

    def get_num_states(self):

        return len(self.__state_frames)

    def get_rep(self, state):

        if state in self.__state_frames:
            return self.__state_frames[state]

        msg = "{} not found in collection.".format(state)
        raise KeyError(msg)

        return

    def save(self, save_loc):
        '''
        Save self to the internal save path. User has no access to this function b/c
        this class is intended to be constructed once and not editied after.
        The save location is the same path as the refCollection the class is init with,
        but with .srep file extension
        '''

        #check if the collection was loaded, and save/load paths are same
        if (self.__was_loaded and self.__load_path == save_loc):

            err_msg = "This save would overwrite the original file at {}".format(self.__load_path)
            raise RuntimeError(err_msg)

        #check if save_loc is a directory, if so, give default name
        if os.path.isdir(save_loc):
            if not save_loc.endswith('/'):
                save_loc += "/"
            save_loc += 'state_database.sframe'

        #if the save_loc doesnt end in sref, append the extension
        if not save_loc.endswith('.sframe'):
            save_loc += '.sframe'

        #pickle self to the outfile location
        with open(save_loc,'wb') as outfile:
            pickle.dump(self, outfile)

        return

    def load(self, load_location):
        #load the StateRefCollection object from the specified folder

        file_found = False

        #determine if a folder or a 'sref' file was provided
        if load_location.endswith(".sframe"):

            self.__load_path = load_location
            file_found = True

        elif os.path.isdir(load_location):

            #search the folder for .sref extensions
            for file in os.listdir(load_location):
                if file.endswith(".sframe"):

                    #check if there are several potential files and give error
                    if (file_found):
                        msg = "Multiple .sframe files found in {}. Specify a single file".format(self.__folder)
                        raise RuntimeError(msg)

                    #set path to file loc and flag that a file was found
                    self.__load_path = os.path.join(load_location, file)
                    file_found = True

        else:

            self.__fnf_error(load_location)
       

        if not file_found:

            self.__fnf_error(load_location)

        #unpickle the file and set the dicts equal
        with open(self.__load_path, 'rb') as f:
            saved_collection = pickle.load(f)

        self.__state_frames = saved_collection.get_dict().copy()

        return


    def __construct_frame_collection(self, refCollection):

        #get the dict of refs from the collection
        refDict = refCollection.get_dict()

        #loop over all states (keys), create a StateRep using the Ref to find the data
        for state in refDict.keys():

            stateRef = refDict[state]
            stateFrame = self.__get_frame(stateRef)
            self.__state_frames[state] = stateFrame

        return


    def __get_frame(self, stateRef):
        #use the data in stateRef to grab the referenced frame

        #get the identifying data
        file  = stateRef.get_file()
        frame = stateRef.get_frame()

        #open the file and frame referenced
        with gsd.hoomd.open(name=file, mode="rb") as snaps:
            snaps = gsd.hoomd.open(name=file, mode="rb")
            L = len(snaps)
            if frame == L:
                frame -= 1
            snap = snaps[frame]

        return snap


    def __set_save_path(self, refCollection):
        #save to the same folder as refCollection was in

        ref_path = refCollection.get_load_path()
        self.__save_path = ref_path.split(".sref")[0] + ".sframe"

        return


    def __add__(self, other_collection):
        #add in all states present in other_collection not in self

        for key,value in other_collection.get_dict().items():

            if key not in self.__state_frames:
                self.__state_frames[key] = value

        return self

    def __fnf_error(self, location):
        #raise a file not found error at the current location

        msg = "Provided load location ({}) could not be found. ".format(location)
        msg+= "Please provide a .sframe file, or folder containing exactly one .sframe file"
        raise FileNotFoundError(msg)
        return


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
