import os
import pytest

#add the paths to the source code folder for testing
import sys
sys.path.insert(0, '../../src')

import numpy as np

from body import body
from body import neighborgrid
from body import cluster

def testBonds():
    #define a simple bond and test it returns the correct values

    type1 = "A"
    type2 = "B"
    cutoff = 10

    bond_name = "A-B"

    ex_bond = body.Bond(type1, type2, cutoff)

    assert(ex_bond.get_types() == tuple((type1,type2)))
    assert(ex_bond.get_cutoff() == cutoff)
    assert(ex_bond.get_name() == bond_name)

    print("Bond Test Passed")

    return

def testNeighborGrid2D():
    #test the neighborgrid on a lattice example

    #define periodic 9x9 grid and create the neighborgrid
    lims = [[0,9],[0,9]]
    R = 1
    periodic = (1,1)
    ng = neighborgrid.Neighborgrid(lims, R, periodic)

    #create test bodies with positions on nodes of lattice
    bodies = []
    for i in range(10):
        for j in range(10):

                b = body.Body(np.array([[1,1]]),[1],1)
                b.set_position(np.array([i,j]))
                bodies.append(b)

    #update the neighbor list with these bodies
    ng.update(bodies)
    test_body = bodies[0]
    # print("Test body located at {}".format(test_body.get_position()))
    # for entry in ng.getNeighborhood(test_body):
    #   print("Neighbor found at {}".format(entry.get_position()))

    assert(len(list(ng.getNeighborhood(test_body))) == 8)

    print("2D Neighborgrid Test Passed")

    return

def testBodyBind():

    #create test bodies using arbitrary data
    particle1_positions = np.array([[1,1],[2,1]])
    particle1_types     = ['A', 'A']
    body1               = body.Body(particle1_positions, particle1_types, 0)

    particle2_positions = np.array([[2,1],[1,2]])
    particle2_types     = ['B', 'B']
    body2               = body.Body(particle2_positions, particle2_types, 1)

    #run a bind check on particles and bind the bodies
    test_particle1 = body1.get_particles()[0]
    test_particle2 = body2.get_particles()[1]
    test_particle2b = body2.get_particles()[0]


    if not test_particle1.get_body().is_bonded(test_particle2.get_body()):
        test_particle1.bind(test_particle2)

    #check if body 1 is binded to body 2 and vise versa
    bf = body1.is_bonded(body2)
    bb = body2.is_bonded(body1)
    assert(bf == True)
    assert(bb == True)

    #get the bond list for body1, check if the entry is body with id 1
    bond_list = body1.get_bond_list()
    assert(bond_list[0].get_id() == 1)

    #try to do the same in reverse to see if it gets stopped
    if not test_particle2b.get_body().is_bonded(test_particle1.get_body()):
        test_particle2b.bind(test_particle1)

    #check that the bond list only has a single entry
    bond_list = body2.get_bond_list()
    assert(bond_list[0].get_id() == 0)
    assert(len(bond_list) == 1)

    print("Body Binding Test Passed")


def testClusteringMerge():

    #init arrays for storage
    old_bodies = []
    cluster_info = []

    #create a list of 10 generic bodies
    bodies     = []
    for i in range(10):
        bodies.append(body.Body([0],[0],i))

    #assign these bodies to two clusters - split 6 and 4
    clusters = []
    clusters.append(cluster.Cluster(bodies[0:4], 0))
    clusters.append(cluster.Cluster(bodies[4:10], 0))

    #do first call on update_clusters - sets the initial clusters with id 0 and 1
    clusters, cluster_info = cluster.update_clusters(clusters, cluster_info, bodies, 
                                                     old_bodies, 0, observer)

    #check that there are two clusters with corresponding indices
    assert(len(cluster_info) == 2)
    assert(cluster_info[0].get_data()[0]['num_bodies'] == 4)
    assert(cluster_info[1].get_data()[0]['num_bodies'] == 6)

    #set old bodies to bodies, make new bodies, assign them all to same cluster
    old_bodies = bodies
    bodies = []
    for i in range(10):
        bodies.append(body.Body([0],[0],i))

    clusters = []
    clusters.append(cluster.Cluster(bodies, 1))

    #call update again
    clusters, cluster_info = cluster.update_clusters(clusters, cluster_info, bodies, 
                                                     old_bodies, 1, observer)

    #do check on clusterinfo
    assert(cluster_info[0].get_data()[0]['num_bodies'] == 4)
    assert(cluster_info[1].get_data()[0]['num_bodies'] == 6)
    assert(cluster_info[0].get_data()[1]['num_bodies'] == 10)
    assert(cluster_info[1].get_data()[1]['num_bodies'] == 10)
    assert(cluster_info[0].is_dead() == True)

    print("Merge Test 1 Passed")

    return



def testClusteringSplit():

    #init arrays for storage
    old_bodies = []
    cluster_info = []

    #create a list of 10 generic bodies
    bodies     = []
    for i in range(10):
        bodies.append(body.Body([0],[0],i))

    #assign bodies to single cluster
    clusters = []
    clusters.append(cluster.Cluster(bodies, 0))

    #do first call on update_clusters - sets the initial cluster with id 0
    clusters, cluster_info = cluster.update_clusters(clusters, cluster_info, bodies, 
                                                     old_bodies, 0, observer)

    #check that there is one cluster with index 0
    assert(len(cluster_info) == 1)
    assert(cluster_info[0].get_data()[0]['num_bodies'] == 10)

    #set old bodies to bodies, make new bodies, assign them split to two clusters
    old_bodies = bodies
    bodies = []
    for i in range(10):
        bodies.append(body.Body([0],[0],i))

    clusters = []
    clusters.append(cluster.Cluster(bodies[0:4], 1))
    clusters.append(cluster.Cluster(bodies[4:10], 1))

    #call update again
    clusters, cluster_info = cluster.update_clusters(clusters, cluster_info, bodies, 
                                                     old_bodies, 1, observer)

    #do check on clusterinfo
    assert(cluster_info[0].get_data()[0]['num_bodies'] == 10)
    assert(cluster_info[1].get_data()[0]['num_bodies'] == 10)
    assert(cluster_info[0].get_data()[1]['num_bodies'] == 6)
    assert(cluster_info[1].get_data()[1]['num_bodies'] == 4)

    print("Split Test 1 Passed")

    return


def testMonomerLoss(num_mon = 1):

    #init arrays for storage
    old_bodies = []
    cluster_info = []

    #create a list of 10 generic bodies
    bodies     = []
    for i in range(10):
        bodies.append(body.Body([0],[0],i))

    #assign bodies to single cluster
    clusters = []
    clusters.append(cluster.Cluster(bodies, 0))

    #do first call on update_clusters - sets the initial cluster with id 0
    clusters, cluster_info = cluster.update_clusters(clusters, cluster_info, bodies, 
                                                     old_bodies, 0, observer)

    #check that there is one cluster with index 0
    assert(len(cluster_info) == 1)
    assert(cluster_info[0].get_data()[0]['num_bodies'] == 10)

    #set old bodies to bodies, make new bodies, remove some as monomers
    old_bodies = bodies
    bodies = []
    for i in range(10):
        bodies.append(body.Body([0],[0],i))

    clusters = []
    clusters.append(cluster.Cluster(bodies[0:10-num_mon], 1))

    #call update again
    clusters, cluster_info = cluster.update_clusters(clusters, cluster_info, bodies, 
                                                     old_bodies, 1, observer)

    # print(cluster_info[0].get_data())

    assert(cluster_info[0].get_data()[0]['num_bodies'] == 10)
    assert(cluster_info[0].get_data()[1]['num_bodies'] == 10-num_mon)

    # print(cluster_info[0].get_monomer_loss_data())

    assert(cluster_info[0].get_monomer_loss_data()[0][0]['num_bodies'] == 10)
    assert(cluster_info[0].get_monomer_loss_data()[0][1] == num_mon)

    print("Monomer Loss Test {} Passed".format(num_mon))

    return


def testMonomerGain(num_mon = 1):

    #init arrays for storage
    old_bodies = []
    cluster_info = []

    #create a list of 10 generic bodies
    bodies     = []
    for i in range(10):
        bodies.append(body.Body([0],[0],i))

    #assign all but num_mon bodies to single cluster
    clusters = []
    clusters.append(cluster.Cluster(bodies[0:10-num_mon], 0))

    #do first call on update_clusters - sets the initial cluster with id 0
    clusters, cluster_info = cluster.update_clusters(clusters, cluster_info, bodies, 
                                                     old_bodies, 0, observer)

    #check that there is one cluster with index 0
    assert(len(cluster_info) == 1)
    assert(cluster_info[0].get_data()[0]['num_bodies'] == 10-num_mon)

    #set old bodies to bodies, make new bodies, make cluster with all bodies
    old_bodies = bodies
    bodies = []
    for i in range(10):
        bodies.append(body.Body([0],[0],i))

    clusters = []
    clusters.append(cluster.Cluster(bodies[0:10], 1))

    #call update again
    clusters, cluster_info = cluster.update_clusters(clusters, cluster_info, bodies, 
                                                     old_bodies, 1, observer)

    # print(cluster_info[0].get_data())

    assert(cluster_info[0].get_data()[0]['num_bodies'] == 10-num_mon)
    assert(cluster_info[0].get_data()[1]['num_bodies'] == 10)

    # print(cluster_info[0].get_monomer_gain_data())

    assert(cluster_info[0].get_monomer_gain_data()[-1][0]['num_bodies'] == 10)
    assert(cluster_info[0].get_monomer_gain_data()[-1][1] == num_mon)

    print("Monomer Gain Test {} Passed".format(num_mon))

    return

def testMonomerGainLoss():
    #test what happens if one monomer attaches and another detaches. 

    #init arrays for storage
    old_bodies = []
    cluster_info = []

    #create a list of 10 generic bodies
    bodies     = []
    for i in range(10):
        bodies.append(body.Body([0],[0],i))

    #assign all but body 0 to single cluster
    clusters = []
    clusters.append(cluster.Cluster(bodies[1:10], 0))

    #do first call on update_clusters - sets the initial cluster with id 0
    clusters, cluster_info = cluster.update_clusters(clusters, cluster_info, bodies, 
                                                     old_bodies, 0, observer)

    #check that there is one cluster with index 0
    assert(len(cluster_info) == 1)
    assert(cluster_info[0].get_data()[0]['num_bodies'] == 9)

    #set old bodies to bodies, make new bodies, make cluster with body0 but w/o 9
    old_bodies = bodies
    bodies = []
    for i in range(10):
        bodies.append(body.Body([0],[0],i))

    clusters = []
    clusters.append(cluster.Cluster(bodies[0:9], 1))

    #call update again
    clusters, cluster_info = cluster.update_clusters(clusters, cluster_info, bodies, 
                                                     old_bodies, 1, observer)

    # print(cluster_info[0].get_data())

    assert(cluster_info[0].get_data()[0]['num_bodies'] == 9)
    assert(cluster_info[0].get_data()[1]['num_bodies'] == 9)

    # print(cluster_info[0].get_monomer_gain_data())

    assert(cluster_info[0].get_monomer_gain_data()[-1][0]['num_bodies'] == 9)
    assert(cluster_info[0].get_monomer_gain_data()[-1][1] == 1)

    # print(cluster_info[0].get_monomer_loss_data())

    assert(cluster_info[0].get_monomer_loss_data()[0][0]['num_bodies'] == 9)
    assert(cluster_info[0].get_monomer_loss_data()[0][1] == 1)

    print("Monomer Gain + Loss Test Passed")
    return


def testDimerization():
    #test two monomers forming a cluster

    #init arrays for storage
    old_bodies = []
    cluster_info = []

    #create a list of 2 generic bodies
    bodies     = []
    for i in range(2):
        bodies.append(body.Body([0],[0],i))

    #only monomers, no clusters
    clusters = []
    observer.current_monomer = 1

    #do first call on update_clusters - sets the initial cluster with id 0
    clusters, cluster_info = cluster.update_clusters(clusters, cluster_info, bodies, 
                                                     old_bodies, 0, observer)

    #check that there are no clusters
    assert(len(cluster_info) == 0)

    #set old bodies to bodies, make new bodies, make cluster with both bodies
    old_bodies = bodies
    bodies = []
    for i in range(2):
        bodies.append(body.Body([0],[0],i))

    clusters = []
    clusters.append(cluster.Cluster(bodies, 1))
    observer.previous_monomer = observer.current_monomer
    observer.current_monomer  = 0

    #call update again
    clusters, cluster_info = cluster.update_clusters(clusters, cluster_info, bodies, 
                                                     old_bodies, 1, observer)

    #check there is a cluster with two bodies, and monomer gain data says 2
    print(cluster_info[0].get_data())

    assert(cluster_info[0].get_data()[0]['num_bodies'] == 2)
    assert(cluster_info[0].get_data()[0]['monomer_fraction'] == 0)

    print(cluster_info[0].get_monomer_gain_data())

    assert(cluster_info[0].get_monomer_gain_data()[0][0]['num_bodies'] == 2)
    assert(cluster_info[0].get_monomer_gain_data()[0][0]['monomer_fraction'] == 1)
    assert(cluster_info[0].get_monomer_gain_data()[0][1] == 2)

    print("Dimerization Test Passed")

    return




if __name__ == "__main__":

    #test setup of classes for body-body interactions
    testBonds()
    testNeighborGrid2D()
    testBodyBind()

    #set a common observer for testing
    observer = cluster.Observer('')
    observer.add_observable('num_bodies')

    #test common clustering scenarios
    testClusteringMerge()
    testClusteringSplit()
    testMonomerLoss()
    testMonomerLoss(2)
    testMonomerGain()
    testMonomerGain(2)
    testMonomerGainLoss()
    testDimerization()
