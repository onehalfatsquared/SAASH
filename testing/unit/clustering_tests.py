import os
import pytest

#add the paths to the source code folder for testing
import sys
sys.path.insert(0, '../../SAASH')

import numpy as np

from structure import body
from structure import cluster
from structure import frame
from util import neighborgrid

import clustering_examples as examples


def testClusteringMerge():

    #get the clusterinfo for the example
    cluster_info = examples.setupExample('merge')
    
    #do check on clusterinfo
    assert(cluster_info[0].get_data()[0]['num_bodies'] == 4)
    assert(cluster_info[1].get_data()[0]['num_bodies'] == 6)
    assert(cluster_info[0].get_data()[1]['num_bodies'] == 10)
    assert(cluster_info[1].get_data()[1]['num_bodies'] == 10)
    assert(cluster_info[0].is_absorbed() == True)

    print("Merge Test 1 Passed")

    return



def testClusteringSplit():

    #get the clusterinfo for the example
    cluster_info = examples.setupExample('split')

    #do check on clusterinfo
    assert(cluster_info[0].get_data()[0]['num_bodies'] == 10)
    assert(cluster_info[1].get_data()[0]['num_bodies'] == 10)
    assert(cluster_info[0].get_data()[1]['num_bodies'] == 6)
    assert(cluster_info[1].get_data()[1]['num_bodies'] == 4)

    print("Split Test 1 Passed")

    return


def testMonomerLoss(num_mon = 1):

    #get the clusterinfo for the example
    cluster_info = examples.setupExample('mon_loss', num_lost = num_mon)

    # print(cluster_info[0].get_data())

    assert(cluster_info[0].get_data()[0]['num_bodies'] == 10)
    assert(cluster_info[0].get_data()[1]['num_bodies'] == 10-num_mon)

    # print(cluster_info[0].get_monomer_loss_data())

    assert(cluster_info[0].get_monomer_loss_data()[1]['num_bodies'] == 10)
    assert(cluster_info[0].get_monomer_loss_data()[1]['num_monomers'] == num_mon)

    print("Monomer Loss Test {} Passed".format(num_mon))

    return


def testMonomerGain(num_mon = 1):

    #get the clusterinfo for the example
    cluster_info = examples.setupExample('mon_gain', num_added = num_mon)

    # print(cluster_info[0].get_data())

    assert(cluster_info[0].get_data()[0]['num_bodies'] == 10-num_mon)
    assert(cluster_info[0].get_data()[1]['num_bodies'] == 10)

    # print(cluster_info[0].get_monomer_gain_data())

    assert(cluster_info[0].get_monomer_gain_data()[1]['num_bodies'] == 10)
    assert(cluster_info[0].get_monomer_gain_data()[1]['num_monomers'] == num_mon)

    print("Monomer Gain Test {} Passed".format(num_mon))

    return

def testMonomerGainLoss():
    #test what happens if one monomer attaches and another detaches. 

    #get the clusterinfo for the example
    cluster_info = examples.setupExample('mon_gain_loss')

    # print(cluster_info[0].get_data())

    assert(cluster_info[0].get_data()[0]['num_bodies'] == 9)
    assert(cluster_info[0].get_data()[1]['num_bodies'] == 9)

    # print(cluster_info[0].get_monomer_gain_data())

    assert(cluster_info[0].get_monomer_gain_data()[1]['num_bodies'] == 9)
    assert(cluster_info[0].get_monomer_gain_data()[1]['num_monomers'] == 1)

    # print(cluster_info[0].get_monomer_loss_data())

    assert(cluster_info[0].get_monomer_loss_data()[1]['num_bodies'] == 9)
    assert(cluster_info[0].get_monomer_loss_data()[1]['num_monomers'] == 1)

    print("Monomer Gain + Loss Test Passed")
    return


def testDimerization():
    #test two monomers forming a cluster

    #get the clusterinfo for the example
    cluster_info = examples.setupExample('dimerization')

    #check there is a cluster with two bodies, and monomer gain data says 2
    # print(cluster_info[0].get_data())

    assert(cluster_info[0].get_data()[0]['num_bodies'] == 2)
    assert(cluster_info[0].get_data()[0]['monomer_fraction'] == 0)

    # print(cluster_info[0].get_monomer_gain_data())

    assert(cluster_info[0].get_monomer_gain_data()[1]['num_bodies'] == 2)
    assert(cluster_info[0].get_monomer_gain_data()[1]['monomer_fraction'] == 1)
    assert(cluster_info[0].get_monomer_gain_data()[1]['num_monomers'] == 2)

    print("Dimerization Test Passed")

    return


def testGetEventsWithLag():
    #test what events get output for a test trajectory at various lags

    cluster_info = examples.setupExample('4step')

    #can do a thorough check of all of the expected outputs
    #main cluster checks
    assert(cluster_info[1].get_transitions(0,1) == [(3,3)])
    assert(cluster_info[1].get_transitions(0,2) == [(3,4),(1,4)])
    assert(cluster_info[1].get_transitions(0,3) == [(3,7),(1,7)])
    assert(cluster_info[1].get_transitions(0,4) == [(3,6),(1,6),(3,1)])
    assert(cluster_info[1].get_transitions(1,1) == [(3,4),(1,4)])
    assert(cluster_info[1].get_transitions(1,2) == [(3,7),(1,7)])
    assert(cluster_info[1].get_transitions(1,3) == [(3,6),(1,6),(3,1)])
    assert(cluster_info[1].get_transitions(2,1) == [(4,7)])
    assert(cluster_info[1].get_transitions(2,2) == [(4,6),(4,1)])
    assert(cluster_info[1].get_transitions(3,1) == [(7,6),(7,1)])

    #side cluster checks
    assert(cluster_info[0].get_transitions(0,1) == [(3,3)])
    assert(cluster_info[0].get_transitions(0,2) == [(3,3)])
    assert(cluster_info[0].get_transitions(0,3) == [(3,7)])
    assert(cluster_info[0].get_transitions(1,1) == [(3,3)])
    assert(cluster_info[0].get_transitions(1,2) == [(3,7)])
    assert(cluster_info[0].get_transitions(2,1) == [(3,7)])

    print("4-Step Test w/ Multiple Lags Passed")

    return

def testMonomerization(n=2):
    #test what events get output for a test trajectory at various lags

    cluster_info = examples.setupExample('monomerization_long', num_lost=n)

    #can do a thorough check of all of the expected outputs

    # print(cluster_info[0].get_data())
    # print(cluster_info[0].get_monomer_loss_data())

    #main cluster checks
    assert(cluster_info[0].get_transitions(0,1) == [(n,n)])
    assert(cluster_info[0].get_transitions(0,2) == [(n,n)])
    assert(cluster_info[0].get_transitions(0,3) == [(n,1)]*n)
    assert(cluster_info[0].get_transitions(1,1) == [(n,n)])
    assert(cluster_info[0].get_transitions(1,2) == [(n,1)]*n)
    assert(cluster_info[0].get_transitions(2,1) == [(n,1)]*n)

    print("Monomerization Test, n={} w/ Multiple Lags Passed".format(n))

    return


def testMultiStepLoss():
    #test events for gradual loss of monomers until dissociation
    cluster_info = examples.setupExample('multi_step_loss')

    # print(cluster_info[0].get_data())
    # print(cluster_info[0].get_monomer_loss_data())

    assert(cluster_info[0].get_transitions(0,1) == [(4,3),(4,1)])
    assert(cluster_info[0].get_transitions(0,2) == [(4,2),(4,1),(4,1)])
    assert(cluster_info[0].get_transitions(0,3) == [(4,1),(4,1),(4,1),(4,1)])
    assert(cluster_info[0].get_transitions(1,1) == [(3,2),(3,1)])
    assert(cluster_info[0].get_transitions(1,2) == [(3,1),(3,1),(3,1)])
    assert(cluster_info[0].get_transitions(2,1) == [(2,1),(2,1)])

    print("Multiple Step Loss Test Passed")
    return






if __name__ == "__main__":

    #test common clustering scenarios
    testClusteringMerge()
    testClusteringSplit()
    testMonomerLoss()
    testMonomerLoss(2)
    testMonomerGain()
    testMonomerGain(2)
    testMonomerGainLoss()
    testDimerization()

    #test counting of events with various lag times
    testGetEventsWithLag()
    testMonomerization(2)
    testMonomerization(3)
    testMonomerization(4)
    testMultiStepLoss()