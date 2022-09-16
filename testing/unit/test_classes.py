import os
import pytest

#add the paths to the relevant folder for testing
import sys
sys.path.insert(0, '../../src')

from body import body
 
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

	print(ex_bond.get_name())

	return









if __name__ == "__main__":

	testBonds()
