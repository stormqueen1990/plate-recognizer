#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import re
import math

# Sizes for the layers
class LayerSizes:
	# Input layer size
	IN = 48
	# Output layer size
	OUT = 6

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
	def train(self, pattFile, trainOutputFile, numIter, learnRate, expectedError, stopIter):
		# Constructs the initial training matrix
		inMiddle = [ [ random.random() for i in range(midLayerSize) ] for i in range(LayerSizes.IN) ]
		middleOut = [ [ random.random() for i in range(LayerSizes.OUT) ] for i in range(midLayerSize) ]

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
		
		# Compute data for each training pair
		for pair in pattPairs:
			itError = 0
			numberIt = 0

			# Train pattern until error is lte expected error
			while itError > expectedError or numberIt > stopIter:
				outputMid = [ None for i in range(midLayerSize) ]
				outputOut = [ None for i in range(LayerSizes.OUT) ]
				delta = None
			
				# Compute outputs for middle layer
				for idx in range(midLayerSize):
					outputMid[idx] = sum(float(pair.inputPatt[idx]) * inMiddle[i][idx] for i in range(LayerSizes.IN))
		
				# Compute outputs for output layer
				for idx in range(LayerSizes.OUT):
					outputOut[idx] = self.__sigFunc(sum(middleOut[i][idx] * outputMid[idx] for i in range(midLayerSize)))

				# Compute errors
				delta = [ (float(pair.expectOut[idx]) - outputOut[idx]) for idx in range(LayerSizes.OUT) ]

				# Catch the greatest error
				for error in delta:
					if error > itError:
						itError = error

				if itError > expectedError:
					for idx in range(LayerSizes.OUT):
						weightAdjust = delta
