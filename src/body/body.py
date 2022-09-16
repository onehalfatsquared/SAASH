'''

This file contains the class implemntations for bodies, particles, and bonds. 

'''

import gsd.hoomd
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

import warnings
import sys
import os


class Nano:

	def __init__(self, ntype, radius, position = None):

		#set the variables given that define a nanoparticle
		self.__position = position
		self.__ntype    = ntype
		self.__radius   = radius

	#getter functions 

	def get_position(self):

		return self.__position

	def get_type(self):

		return self.__ntype

	def get_radius(self):

		return self.__radius

class Bond:

	def __init__(self, type1, type2, cutoff):

		#set the variables given that define the bond
		self.__type1  = type1
		self.__type2  = type2
		self.__cutoff = cutoff

		#give the bond a descriptive strign name - "type1-type2"
		self.__bond_name = type1 + "-" + type2

	#getter functions

	def get_types(self):

		return tuple((self.__type1, self.__type2))

	def get_cutoff(self):

		return self.__cutoff

	def get_name(self):

		return self.__bond_name



class Particle:

	def __init__(self, position, p_type, body_id):

		#set the variables given in constructor
		self.__position = position
		self.__p_type   = p_type
		self.__body_id  = body_id

	#getter functions 

	def get_position(self):

		return self.__position

	def get_type(self):

		return self.__p_type

	def get_body(self):

		return __body_id



class Body:

	def __init__(self, ):

		self.__body_id = 0


