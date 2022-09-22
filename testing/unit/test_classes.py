import os
import pytest

#add the paths to the relevant folder for testing
import sys
sys.path.insert(0, '../../src')

import numpy as np

from body import body
from body import neighborgrid
 
# tests start here

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

	return

def testNeighborGrid():
	#test the neighborgrid on a lattice example

	#define periodic 9x9x9 grid and create the neighborgrid
	lims = [[0,9],[0,9]]
	R = 1
	periodic = (1,1)
	ng = neighborgrid.Neighborgrid(lims, R, periodic)

	#create test bodies with positions on nodes of lattice
	bodies = []
	for i in range(10):
		for j in range(10):

				b = body.Body(np.array([[1,1]]),[1],1)
				b.set_position((i,j))
				bodies.append(b)

	#update the neighbor list with these bodies
	ng.update(bodies)
	test_body = bodies[34]
	# print("Test body located at {}".format(test_body.get_position()))
	# for entry in ng.getNeighborhood(test_body):
	# 	print("Neighbor found at {}".format(entry.get_position()))

	assert(len(list(ng.getNeighborhood(test_body))) == 6)

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





#Todo - write a test to check if body A binds to body B, if the reverse check works correctly














if __name__ == "__main__":

	testBonds()
	testNeighborGrid()
	testBodyBind()
