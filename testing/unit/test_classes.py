import os
import pytest

#add the paths to the source code folder for testing
import sys
sys.path.insert(0, '../../SAASH')

import numpy as np
import matplotlib.pyplot as plt

from structure import body
from structure import cluster
from structure import frame
from util import neighborgrid


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
    R = 1.5
    periodic = (1,1)
    ng = neighborgrid.Neighborgrid(lims, R, periodic)

    #create test bodies with positions on nodes of lattice
    bodies = []
    for i in range(9):
        for j in range(9):

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

def testNeighborGrid3D():

    #define periodic 9x9x9 grid and create the neighborgrid
    lims = [[0,9],[0,9],[0,9]]
    R = 1.5
    periodic = (1,1,1)
    ng = neighborgrid.Neighborgrid(lims, R, periodic)

    #create test bodies with positions on nodes of lattice
    bodies = []
    for i in range(9):
        for j in range(9):
            for k in range(9):

                b = body.Body(np.array([[1,1,1]]),[1],1)
                b.set_position(np.array([i,j,k]))
                bodies.append(b)

    #update the neighbor list with these bodies
    ng.update(bodies)
    test_body = bodies[17]
    # print("Test body located at {}".format(test_body.get_position()))
    # for entry in ng.getNeighborhood(test_body):
    #   print("Neighbor found at {}".format(entry.get_position()))

    # print(len(list(ng.getNeighborhood(test_body))))
    assert(len(list(ng.getNeighborhood(test_body))) == 18)

    print("3D Neighborgrid Test Passed")
    return

def plotNGpoints():

    R = np.linspace(0.5,2.1,150)
    values = []

    for i in range(150):

        lims = [[0,9],[0,9]]
        Ri = R[i]
        periodic = (1,1)
        ng = neighborgrid.Neighborgrid(lims, Ri, periodic)

        #create test bodies with positions on nodes of lattice
        bodies = []
        for i in range(9):
            for j in range(9):

                b = body.Body(np.array([[1,1]]),[1],1)
                b.set_position(np.array([i,j]))
                bodies.append(b)

        #update the neighbor list with these bodies
        ng.update(bodies)
        test_body = bodies[0]
        values.append(len(list(ng.getNeighborhood(test_body))))

    plt.plot(R, values)
    plt.show()
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
        test_particle1.bind(test_particle2, body.Bond('A','B',1))

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
        test_particle2b.bind(test_particle1, body.Bond('A','B',1))

    #check that the bond list only has a single entry
    bond_list = body2.get_bond_list()
    assert(bond_list[0].get_id() == 0)
    assert(len(bond_list) == 1)

    print("Body Binding Test Passed")

    return







if __name__ == "__main__":

    #test setup of classes for body-body interactions
    testBonds()
    testNeighborGrid2D()
    testNeighborGrid3D()
    testBodyBind()

    # plotNGpoints()
