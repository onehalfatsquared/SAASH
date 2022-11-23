'''

An Observer class for the Self-Assembly analysis. This class stores what kind of 
information the user is looking to compute during the analysis. Options include...

1) bulk

This mode identifies the number of clusters of each size, as well as the largest 
cluster, in every frame. Output as a .cl file

2) nanoparticle

This mode only considers assembly in the vicinity of nanoparticles. Will output
cluster properties that are set in the observer for each nanoparticle in a .np file. 

3) cluster

This mode will treat each cluster (2 or more bonded particles) seperately. Will track 
all properties specified in the observer across each cluster's lifetime. Automatically
includes the monomer fraction in this list of data. Will be pickled as a .pkl file. 


'''

import numpy as np
import sys




class Observer:

    def __init__(self, gsd_file = None, run_type = None):

        #print init message
        print("\nConstructing an Observer")

        #set the allowed run type options and corresponding outfile extensions
        self.__allowed_run_types = ['bulk', 'nanoparticle', 'cluster']
        self.__file_extensions   = {'bulk':'.cl', 'nanoparticle':'.np', 'cluster':'.pkl'}

        #init a set of observables to compute
        self.__observable_set = set()


        #init variable to store the runtype
        self.__run_type = None
        if run_type:
            self.set_run_type(run_type)

        #use the gsd file to determine an output file for this run
        self.__gsd_file = gsd_file
        self.__outfile  = None
        if self.__gsd_file and self.__run_type:
            self.__set_outfile_name(gsd_file)





    def add_observable(self, observable):

        self.__observable_set.add(observable)
        print("Observable {} added to observer".format(observable))
        return

    def get_observables(self):

        return self.__observable_set

    def get_outfile(self):

        return self.__outfile

    def get_run_type(self):

        return self.__run_type

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

    def compute_coordinate(self, cluster):
        '''this computes various observables for the cluster, based on user input
           given to the observer class. Default is simply number of bodies'''

        #init a dict to store properties of the cluster
        property_dict = dict()

        #get the list of observables requested, and compute them from cluster
        observables = self.get_observables()
        for obs in observables:

            if obs == "num_bodies":

                property_dict['num_bodies'] = len(cluster.get_bodies())

            elif obs == "positions":

                property_dict['positions'] = cluster.get_body_positions()

            else:

                raise("The requested property is not implemented. Check that it is"\
                    " implemented and spelled correctly")


            

        return property_dict

    def __set_outfile_name(self, gsd_file):
        #use the prefix up to '.gsd' to set a file path for the output of this analysis

        prefix = gsd_file.split('.gsd')[0]
        self.__outfile = prefix + self.__file_extensions[self.__run_type]
        print("Output file set as: {}\n".format(self.__outfile))

        return
