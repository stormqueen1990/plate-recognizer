#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import re

# Sizes for the layers
class NetSizes:
	# Input layer size
	IN_LAYER = 48
	# Output layer size
	OUT_LAYER = 6

# Link identifiers
class LinkIdent:
	# Between input and middle layers
	IN_MIDDLE = 'IM'
	# Between middle and output layers
	MIDDLE_OUT = 'MO'

# Operation type
class OpType:
	TRAIN = 1
	RECOG = 2

# Training type
class TrainType:
	BATCH = 1
	ALTERNATED = 2

# Pattern pairs class
class PattPair:
	def __init__(self, inputPatt, expectOut):
		self.inputPatt = inputPatt
		self.expectOut = expectOut

# Plate recognizement kernel
class PlateRecognizer:
	# Trains a randomly constructed neural net
	def train(self, pattFile, trainOutputFile, numIter, ):
		# Constructs the initial training matrix
		inMiddle = [ [ random.random() for i in range(midLayerSize) ] for i in range(IN_LAYER) ]
		middleOut = [ [ random.random() for i in range(OUT_LAYER) ] for i in range(midLayerSize) ]
		net = { IN_MIDDLE : inMiddle, MIDDLE_OUT : middleOut }

		# List with all pairs
		pattPairs = []

		# Creates the regexp to search for patterns
		regexpPatt = re.compile("([01]+)")

		# Searches the file for patterns
		fPatt = open(pattFile, "r")

		for line in fPatt:
			it = fPatt.findall(line)

			# Does line contain a pair?
			if len(it) < 2:
				raise ValueError("all lines must contain a pair")

			p = PattPair(it[0], it[1])
			pattPairs.append(p)

		for pair in pattPairs:
			for bit in pair.inputPatt:

