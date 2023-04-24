'''

This script takes a folder that contains BOTH a .gsd AND a .cl file from a simulation. 
It determines all the unique states that were visited, creates a reference to where that 
state can be found in the gsd file, and then extracts the required coordinates to reconstruct 
that state and stores it in a dictionary.

'''

from SAASH.util.state import StateScraper
from SAASH.util.state import StateRefCollection
from SAASH.util.state import StateRepCollection
from SAASH.util.state import StateRef
from SAASH.util.state import State

import os, sys
import pickle

#CRITICAL: If things are pickled within the util module of SAASH, 
#the following is required to unpickle outside of that project's namespace
from SAASH import util
sys.modules['util'] = util


def create_sref(folder):
    #Create a StateRefCollection or load an existing one

    #scrape the cluster file for unique states - creates .sref file
    print("Gathering unique states...")
    StateScraper(folder, verbose=True)

    #load and return the collection
    print("Creating State Reference Collection...")
    reference_collection = StateRefCollection(folder)
    return reference_collection


def create_srep(reference_collection):
    #create a StateRepCollection from the reference data

    print("Creating State Representation Collection...")
    StateRepCollection(reference_collection)
    return

    

if __name__ == "__main__":
    
    try:
        folder = sys.argv[1]
    except:
        print("Usage: %s <trajectory_folder>" % sys.argv[0])
        raise

    reference_collection = create_sref(folder)
    create_srep(reference_collection)
        
