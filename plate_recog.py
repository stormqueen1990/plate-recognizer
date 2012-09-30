#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import re
import math

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
	# Sigmoid function
	def __sigFunc(self, sumVal):
		return 1.0 / (1.0 + math.exp(-1.0 * sumVal));

	# Trains a randomly constructed neural net
	def train(self, pattFile, trainOutputFile, numIter, learnTax, expectedError):
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

		itError = 0
		convIter = 0
		while numIter > 0:
			for pair in pattPairs:
				output = [ None for i in range(len(pattPairs)) ]
				error = [ None for i in range(len(pattPairs)) ]
				idx = 0

				for bit in pair.inputPatt:
					output[idx] = sum(bit * weight for weight in net[IN_MIDDLE][idx]))
					idx = idx + 1
			
				idx = 0
				for val in output:
					output[idx] = self.__sigFunc(sum(val * weight for weight in net[MIDDLE_OUT][idx]))
					idx = idx + 1

			numIter = numIter - 1
