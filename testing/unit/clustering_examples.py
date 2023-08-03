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
from util import observer as obs

#set a common observer for testing
observer = obs.Observer('')
observer.init_default_set()
observer.set_run_type('cluster')
observer.set_outfile('example.gsd')


####################################################################
############ Setup for clustering examples go here  ################
####################################################################

def setup4step():
    #example trajectory for 3->3->4->7->6

    #init arrays for storage
    old_bodies = []
    cluster_info = []

    #create a list of 10 generic bodies
    bodies     = []
    for i in range(7):
        bodies.append(body.Body([0],[0],i))

    #assign these bodies to clusters
    clusters = []
    clusters.append(cluster.Cluster(bodies[0:3], 0))
    clusters.append(cluster.Cluster(bodies[3:6], 0))

    #create the first frame with this info
    f0 = frame.Frame(bodies, clusters, 0, [6], 1.0/7.0)
    f0.create_first_frame(cluster_info, 0, observer)

    #update 1
    old_bodies = bodies
    bodies = []
    for i in range(7):
        bodies.append(body.Body([0],[0],i))

    clusters = []
    clusters.append(cluster.Cluster(bodies[0:3], 1))
    clusters.append(cluster.Cluster(bodies[3:6], 1))

    #update frame
    f1 = frame.Frame(bodies, clusters, 1, [6], 1.0/7.0)
    f1.update(cluster_info, f0, observer)

    #update 2
    old_bodies = bodies
    bodies = []
    for i in range(7):
        bodies.append(body.Body([0],[0],i))

    clusters = []
    clusters.append(cluster.Cluster(bodies[0:3], 2))
    clusters.append(cluster.Cluster(bodies[3:7], 2))

    #update frame
    f2 = frame.Frame(bodies, clusters, 2, [], 0)
    f2.update(cluster_info, f1, observer)

    #update 3
    old_bodies = bodies
    bodies = []
    for i in range(7):
        bodies.append(body.Body([0],[0],i))

    clusters = []
    clusters.append(cluster.Cluster(bodies[0:7], 3))

    #update frame
    f3 = frame.Frame(bodies, clusters, 3, [], 0)
    f3.update(cluster_info, f2, observer)

    #update 3
    old_bodies = bodies
    bodies = []
    for i in range(7):
        bodies.append(body.Body([0],[0],i))

    clusters = []
    clusters.append(cluster.Cluster(bodies[0:6], 3))

    #update frame
    f4 = frame.Frame(bodies, clusters, 4, [7], 1.0/7.0)
    f4.update(cluster_info, f3, observer)

    return cluster_info

def setupMerge():
    #example trajectory for 4+6->10

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

    #create the first frame with this info
    f0 = frame.Frame(bodies, clusters, 0, [], 0)
    f0.create_first_frame(cluster_info, 0, observer)

    #set old bodies to bodies, make new bodies, assign them all to same cluster
    old_bodies = bodies
    bodies = []
    for i in range(10):
        bodies.append(body.Body([0],[0],i))

    clusters = []
    clusters.append(cluster.Cluster(bodies, 1))

    #create next frame and do an update
    f1 = frame.Frame(bodies, clusters, 1, [], 0)
    f1.update(cluster_info, f0, observer)

    return cluster_info


def setupSplit():
    #example trajectory for 10->6+4

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

    #create the first frame with this info
    f0 = frame.Frame(bodies, clusters, 0, [], 0)
    f0.create_first_frame(cluster_info, 0, observer)

    #set old bodies to bodies, make new bodies, assign them split to two clusters
    old_bodies = bodies
    bodies = []
    for i in range(10):
        bodies.append(body.Body([0],[0],i))

    clusters = []
    clusters.append(cluster.Cluster(bodies[0:4], 1))
    clusters.append(cluster.Cluster(bodies[4:10], 1))

    #create next frame and do an update
    f1 = frame.Frame(bodies, clusters, 1, [],0)
    f1.update(cluster_info, f0, observer)

    return cluster_info

def setupMonLoss(num_mon = 1):
    #example trajectory for 10->10-num_mon via monomer loss

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

    #create the first frame with this info
    f0 = frame.Frame(bodies, clusters, 0, [], 0)
    f0.create_first_frame(cluster_info, 0, observer)

    #set old bodies to bodies, make new bodies, remove some as monomers
    old_bodies = bodies
    bodies = []
    for i in range(10):
        bodies.append(body.Body([0],[0],i))

    clusters = []
    clusters.append(cluster.Cluster(bodies[0:10-num_mon], 1))

    #call update again
    #create next frame and do an update
    f1 = frame.Frame(bodies, clusters, 1, [9], num_mon/10)
    f1.update(cluster_info, f0, observer)

    return cluster_info

def setupMonGain(num_mon = 1):
    #example trajectory for 10-num_mon->10 via monomer addition

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

    f0 = frame.Frame(bodies, clusters, 0, [10], num_mon/10)
    f0.create_first_frame(cluster_info, 0, observer)

    #set old bodies to bodies, make new bodies, make cluster with all bodies
    old_bodies = bodies
    bodies = []
    for i in range(10):
        bodies.append(body.Body([0],[0],i))

    clusters = []
    clusters.append(cluster.Cluster(bodies[0:10], 1))

    #call update again
    f1 = frame.Frame(bodies, clusters, 1, [], 0)
    f1.update(cluster_info, f0, observer)

    return cluster_info

def setupMonGainLoss():
    #example trajectory for 10->10 but with 1 monomer gained and 1 lost

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
    f0 = frame.Frame(bodies, clusters, 0, [0], 0.1)
    f0.create_first_frame(cluster_info, 0, observer)

    #set old bodies to bodies, make new bodies, make cluster with body0 but w/o 9
    old_bodies = bodies
    bodies = []
    for i in range(10):
        bodies.append(body.Body([0],[0],i))

    clusters = []
    clusters.append(cluster.Cluster(bodies[0:9], 1))

    #call update again
    f1 = frame.Frame(bodies, clusters, 1, [10], 0.1)
    f1.update(cluster_info, f0, observer)

    return cluster_info

def setupDimerization():
    #trajectory example for 1+1->2

    #init arrays for storage
    old_bodies = []
    cluster_info = []

    #create a list of 2 generic bodies
    bodies     = []
    for i in range(2):
        bodies.append(body.Body([0],[0],i))

    #only monomers, no clusters
    clusters = []

    #do first call on update_clusters - sets the initial cluster with id 0
    f0 = frame.Frame(bodies, clusters, 0, [0,1], [0,0], 1)
    f0.create_first_frame(cluster_info, 0, observer)

    #set old bodies to bodies, make new bodies, make cluster with both bodies
    old_bodies = bodies
    bodies = []
    for i in range(2):
        bodies.append(body.Body([0],[0],i))

    clusters = []
    clusters.append(cluster.Cluster(bodies, 1))

    #call update again
    f1 = frame.Frame(bodies, clusters, 1, [], [], 0)
    f1.update(cluster_info, f0, observer)

    return cluster_info

def setupMonomerizationLong(n = 2):
    #trajectory example for n->n->n->1

    #init arrays for storage
    old_bodies = []
    cluster_info = []

    #create a list of 2 generic bodies
    bodies     = []
    for i in range(n):
        bodies.append(body.Body([0],[0],i))

    #only monomers, no clusters
    clusters = []
    clusters.append(cluster.Cluster(bodies, 0))

    #do first call on update_clusters
    f0 = frame.Frame(bodies, clusters, 0, [], 0)
    f0.create_first_frame(cluster_info, 0, observer)

    #update 1
    old_bodies = bodies
    bodies = []
    for i in range(n):
        bodies.append(body.Body([0],[0],i))

    clusters = []
    clusters.append(cluster.Cluster(bodies, 1))

    #update frame
    f1 = frame.Frame(bodies, clusters, 1, [], 0)
    f1.update(cluster_info, f0, observer)

    #update 2
    old_bodies = bodies
    bodies = []
    for i in range(n):
        bodies.append(body.Body([0],[0],i))

    clusters = []
    clusters.append(cluster.Cluster(bodies, 2))

    #update frame
    f2 = frame.Frame(bodies, clusters, 2, [], 0)
    f2.update(cluster_info, f1, observer)

    #update 3
    old_bodies = bodies
    bodies = []
    for i in range(n):
        bodies.append(body.Body([0],[0],i))

    clusters = []

    #update frame
    f3 = frame.Frame(bodies, clusters, 3, range(n), 1)
    f3.update(cluster_info, f2, observer)

    return cluster_info

def setupMultiStepLoss():
    #trajectory example for 4->3->2->1 via monomer loss

    #init arrays for storage
    old_bodies = []
    cluster_info = []

    #create a list of 2 generic bodies
    bodies     = []
    for i in range(4):
        bodies.append(body.Body([0],[0],i))

    #only monomers, no clusters
    clusters = []
    clusters.append(cluster.Cluster(bodies, 0))

    #do first call on update_clusters
    f0 = frame.Frame(bodies, clusters, 0, [], 0)
    f0.create_first_frame(cluster_info, 0, observer)

    #update 1
    old_bodies = bodies
    bodies = []
    for i in range(4):
        bodies.append(body.Body([0],[0],i))

    clusters = []
    clusters.append(cluster.Cluster(bodies[0:3], 1))

    #update frame
    f1 = frame.Frame(bodies, clusters, 1, [3], 0.25)
    f1.update(cluster_info, f0, observer)

    #update 2
    old_bodies = bodies
    bodies = []
    for i in range(4):
        bodies.append(body.Body([0],[0],i))

    clusters = []
    clusters.append(cluster.Cluster(bodies[0:2], 1))

    #update frame
    f2 = frame.Frame(bodies, clusters, 2, [2,3], 0.5)
    f2.update(cluster_info, f1, observer)

    #update 3
    old_bodies = bodies
    bodies = []
    for i in range(4):
        bodies.append(body.Body([0],[0],i))

    clusters = []

    #update frame
    f3 = frame.Frame(bodies, clusters, 3, [0,1,2,3], 1)
    f3.update(cluster_info, f2, observer)

    return cluster_info


def setupMonomerSwap():
    #example trajectory for [(0,1,2),(3,4)] -> [(0,1),(2,3,4)]

    #init arrays for storage
    old_bodies = []
    cluster_info = []

    #create a list of 2 generic bodies
    bodies     = []
    for i in range(5):
        bodies.append(body.Body([0],[0],i))

    #only monomers, no clusters
    clusters = []
    clusters.append(cluster.Cluster(bodies[0:3], 0))
    clusters.append(cluster.Cluster(bodies[3:5], 0))

    #do first call on update_clusters
    f0 = frame.Frame(bodies, clusters, 0, [], 0)
    f0.create_first_frame(cluster_info, 0, observer)

    #update 1
    old_bodies = bodies
    bodies = []
    for i in range(5):
        bodies.append(body.Body([0],[0],i))

    clusters = []
    clusters.append(cluster.Cluster(bodies[0:2], 0))
    clusters.append(cluster.Cluster(bodies[2:5], 0))

    #update frame
    f1 = frame.Frame(bodies, clusters, 1, [], 0)
    f1.update(cluster_info, f0, observer)

    return cluster_info

def setupMonomerSwap2():
    #example trajectory for [(0,1),2,3] -> [(0,2),(1,3)]

    #init arrays for storage
    old_bodies = []
    cluster_info = []

    #create a list of 2 generic bodies
    bodies     = []
    for i in range(4):
        bodies.append(body.Body([0],[0],i))

    #only monomers, no clusters
    clusters = []
    clusters.append(cluster.Cluster(bodies[0:2], 0))

    #do first call on update_clusters
    f0 = frame.Frame(bodies, clusters, 0, [2,3], 0.5)
    f0.create_first_frame(cluster_info, 0, observer)

    #update 1
    old_bodies = bodies
    bodies = []
    for i in range(4):
        bodies.append(body.Body([0],[0],i))

    clusters = []
    clusters.append(cluster.Cluster(bodies[0:3:2], 0))
    clusters.append(cluster.Cluster(bodies[1:4:2], 0))

    #update frame
    f1 = frame.Frame(bodies, clusters, 1, [], 0.0)
    f1.update(cluster_info, f0, observer)

    return cluster_info


def setupM2D2M():
    #example trajectory for 2 monomers to dimer to 2 monomers

    #init arrays for storage
    old_bodies = []
    cluster_info = []

    #create a list of 2 generic bodies
    bodies     = []
    for i in range(2):
        bodies.append(body.Body([0],[0],i))

    #only monomers, no clusters
    clusters = []

    #do first call on update_clusters
    f0 = frame.Frame(bodies, clusters, 0, [0,1], 1)
    f0.create_first_frame(cluster_info, 0, observer)

    #update 1
    old_bodies = bodies
    bodies = []
    for i in range(2):
        bodies.append(body.Body([0],[0],i))

    clusters = []
    clusters.append(cluster.Cluster(bodies, 1))

    #update frame
    f1 = frame.Frame(bodies, clusters, 1, [], 0.0)
    f1.update(cluster_info, f0, observer)

    #update 2
    old_bodies = bodies
    bodies = []
    for i in range(2):
        bodies.append(body.Body([0],[0],i))

    clusters = []

    #update frame
    f2 = frame.Frame(bodies, clusters, 2, [0,1], 1.0)
    f2.update(cluster_info, f1, observer)

    return cluster_info

def setupDDM():
    #example trajectory for 2 dimers and a monomer to pentamer

    #init arrays for storage
    old_bodies = []
    cluster_info = []

    #create a list of 2 generic bodies
    bodies     = []
    for i in range(5):
        bodies.append(body.Body([0],[0],i))

    #two dimers and a monomer
    clusters = []
    clusters.append(cluster.Cluster(bodies[0:2], 0))
    clusters.append(cluster.Cluster(bodies[2:4], 0))

    #do first call on update_clusters
    f0 = frame.Frame(bodies, clusters, 0, [4], 0.2)
    f0.create_first_frame(cluster_info, 0, observer)

    #update 1
    old_bodies = bodies
    bodies = []
    for i in range(5):
        bodies.append(body.Body([0],[0],i))

    clusters = []
    clusters.append(cluster.Cluster(bodies, 1))

    #update frame
    f1 = frame.Frame(bodies, clusters, 1, [], 0.0)
    f1.update(cluster_info, f0, observer)

    return cluster_info







def setupExample(which, num_added = 0, num_lost = 0):
    #setup example trajectories based on the input in which

    if (which == '4step'):

        #example trajectory for 3->3->4->7->6, where 4->7 is trimer addition
        return setup4step()

    elif (which == 'merge'):

        #example trajectory for 4+6->10
        return setupMerge()

    elif (which == 'split'):

        #example trajectory for 10->4+6
        return setupSplit()

    elif (which == 'mon_loss'):

        #example trajectory for 10->10-num_mon via monomer loss
        return setupMonLoss(num_lost)

    elif (which == 'mon_gain'):

        #example trajectory for 10-num_mon->10 via monomer addition
        return setupMonGain(num_added)

    elif (which == 'mon_gain_loss'):

        #example trajectory for 10->10 but with 1 monomer gained and 1 lost
        return setupMonGainLoss()

    elif (which == "dimerization"):

        #example traj for 1+1->2
        return setupDimerization()

    elif (which == "monomerization_long"):

        #example traj for 2->2->2->1
        return setupMonomerizationLong(num_lost)

    elif (which == "multi_step_loss"):

        #example traj for 4->3->2->1 via monomer loss
        return setupMultiStepLoss()

    elif (which == "monomer_swap"):

        #example trajectory for [(0,1,2),(3,4)] -> [(0,1),(2,3,4)]
        return setupMonomerSwap()

    elif (which == "monomer_swap_2"):

        #example trajectory for [(0,1),2,3] -> [(0,2),(1,3)]
        return setupMonomerSwap2()

    elif (which == "m2d2m"):

        #example trajectory for 2 monomers to dimer to 2 monomers
        return setupM2D2M()

    elif (which == "ddm"):

        #example trajectory for 2 dimers and monomer forming 5mer
        return setupDDM()

    else:

        raise("The specified test, {},  has not been implemented".format(which))


