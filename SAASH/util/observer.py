'''

An Observer class for the Self-Assembly analysis. This class stores what kind of 
information the user is looking to compute during the analysis. Options include...

1) bulk

This mode identifies the number of clusters of each size, as well as the largest 
cluster, in every frame. Output as a .sizes file

Optionally, can supply a 'focus list' which will further identify the number of each
microstate of that size (fixed to be number of each type of bond for now)

2) nanoparticle

This mode only considers assembly in the vicinity of nanoparticles. Will output
cluster properties that are set in the observer for each nanoparticle in a .np file. 

3) cluster

This mode will treat each cluster (2 or more bonded particles) seperately. Will track 
all properties specified in the observer across each cluster's lifetime. Automatically
includes the monomer fraction in this list of data. Will be pickled as a .cl file. 


'''

import numpy as np
import sys




class Observer:

    def __init__(self, gsd_file = None, run_type = None, jump=1):

        #print init message
        print("\nConstructing an Observer")

        #set the allowed run type options and corresponding outfile extensions
        self.__allowed_run_types = ['bulk', 'nanoparticle', 'cluster']
        self.__file_extensions   = {'bulk':'.sizes', 'nanoparticle':'.np', 'cluster':'.cl'}

        #init a set of observables to compute
        self.__observable_set = set()

        #define common identifying observables for clusters
        self.__identifying_set = set(['bonds', 'types'])

        #init defaults for frames
        self.__first_frame = 0
        self.__final_frame = None
        self.__jump        = jump

        #set the neighborgrid default cutoff to None, can be overwritten
        self.__ngrid_R = None

        #init variable to store the runtype
        self.__run_type = None
        if run_type:
            self.set_run_type(run_type)

        #use the gsd file to determine an output file for this run
        self.__gsd_file = gsd_file
        self.__outfile  = None
        if self.__gsd_file and self.__run_type:
            self.__set_outfile_name(gsd_file)


        #init a focus list. during a bulk run 
        self.__focus_list = None




    def add_observable(self, observable):

        self.__observable_set.add(observable)
        print("Observable {} added to observer".format(observable))
        return

    def set_focus_list(self, focus_list):

        self.__focus_list = focus_list
        return

    def get_focus_list(self):

        return self.__focus_list

    def get_observables(self):

        return self.__observable_set

    def get_non_trivial_observables(self):

        disallowed = ["num_bodies", "monomer_fraction", 'positions', 'orientations', 'indices']
        nt_observables = [obs for obs in self.__observable_set if obs not in disallowed]
        return nt_observables

    def get_outfile(self):

        return self.__outfile

    def get_run_type(self):

        return self.__run_type

    def get_first_frame(self):

        return self.__first_frame

    def get_final_frame(self):

        return self.__final_frame

    def get_frame_jump(self):

        return self.__jump

    def get_ngrid_cutoff(self):

        return self.__ngrid_R

    def set_ngrid_cutoff(self, ngrid_R):
        #manually set the value for the neighborgrid cutoff distance
        #overwrites the default calculation

        self.__ngrid_R = ngrid_R
        return


    def set_first_frame(self, first_frame):

        if (first_frame < 0):
            raise ValueError("Starting frame ({}) must be non-negative.".format(first_frame))

        self.__first_frame = first_frame

        print("First frame to analyze set to {}".format(first_frame))
        return

    def set_final_frame(self, final_frame):

        if (final_frame < self.__first_frame):
            raise ValueError("Final frame ({}) must be greater than first frame ({})".format(
                              final_frame, self.__first_frame))
        self.__final_frame = final_frame

        print("Final frame to analyze set to {}".format(final_frame))
        return


    def init_test_set(self):
        #init the observable set to those helpful for testing

        self.__observable_set = set(['positions'])
        self.set_run_type('cluster')
        if self.__gsd_file and not self.__out_file:
            self.set_outfile(self.__gsd_file)

        return

    def init_default_set(self):
        #init a test set that just includes num_bodies

        self.__observable_set = set(['num_bodies'])
        self.set_run_type('cluster')
        if self.__gsd_file and not self.__outfile:
            self.set_outfile(self.__gsd_file)

        return

    def set_run_type(self, run_type):
        #set the runtype to be one of a few options

        #check that the run type is supported
        if run_type not in self.__allowed_run_types:
            raise ValueError('The specified run type is not in the list of supported types\n'\
                             +'Allowed types: {}'.format(self.__allowed_run_types))

        #valid run type, so set it

        #check if there is an existing type. if not set it
        if self.__run_type is None:
            self.__run_type = run_type
            print("Run type set as:    {}".format(self.__run_type))

        #determine if the type is being changed. let user know swap occurred
        else:
            if self.__run_type != run_type:
                print("Run type swapped from {} to {}".format(self.__run_type, run_type))
                self.__run_type = run_type


        return

    def set_outfile(self, gsd_input):
        #manually set the outfile name

        self.__set_outfile_name(gsd_input)

        return 

    def compute_observables(self, cluster):
        '''this computes various observables for the cluster, based on user input
           given to the observer class. Default is simply number of bodies'''

        #init a dict to store properties of the cluster
        property_dict = dict()

        #get the list of observables requested, and compute them from cluster
        observables = self.get_observables()
        for obs in observables:
            property_dict[obs] = self.compute_observable(cluster, obs)

        return property_dict

    def compute_observable(self, cluster, obs):
        #compute the observable specified by obs on the cluster

        if obs == "num_bodies":

            return len(cluster.get_bodies())

        elif obs == "positions":

            return cluster.get_body_positions()

        elif obs == "bonds":

            return cluster.get_bond_types()

        elif obs == "indices":

            return cluster.get_body_ids()

        elif obs == "types":

            return cluster.get_body_types()

        else:

            raise("The requested property is not implemented. Check that it is"\
                " implemented and spelled correctly")

        return


    def get_identifying_observables(self):
        #return the active observables that are in the common identifying set

        common = [obs for obs in self.get_observables() if obs in self.__identifying_set]
        return common

    def __set_outfile_name(self, gsd_file):
        #use the prefix up to '.gsd' to set a file path for the output of this analysis

        prefix = gsd_file.split('.gsd')[0]
        self.__outfile = prefix + self.__file_extensions[self.__run_type]
        print("Output file set as: {}\n".format(self.__outfile))

        return
