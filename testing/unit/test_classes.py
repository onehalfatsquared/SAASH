import os
import pytest

#add the paths to the relevant folder for testing
import sys
sys.path.insert(0, '../../src')

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

				b = body.Body()
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













if __name__ == "__main__":

	testBonds()
	testNeighborGrid()
