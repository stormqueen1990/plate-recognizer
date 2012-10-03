#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import re
import math
from decimal import Decimal

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
		one = Decimal(1.0)
		return one / (one + Decimal(Decimal(-sumVal).exp()));

	# Trains a randomly constructed neural net
	def train(self, pattFile, trainOutputFile, midLayerSize, learnRate, expectedError, stopIter):
		# Constructs the initial training matrix
		inMiddle = [ [ Decimal(random.random()) for i in range(midLayerSize) ] for i in range(LayerSizes.IN) ]
		middleOut = [ [ Decimal(random.random()) for i in range(LayerSizes.OUT) ] for i in range(midLayerSize) ]

		# List with all pairs
		pattPairs = []

		# Creates the regexp to search for patterns
		regexpPatt = re.compile("([01]+)")

		# Searches the file for patterns
		fPatt = open(pattFile, "r")

		for line in fPatt:
			it = regexpPatt.findall(line)

			# Does line contain a pair?
			if len(it) < 2:
				raise ValueError("all lines must contain a pair")

			p = PattPair(it[0], it[1])
			pattPairs.append(p)
		
		# Train pattern for x iterations
		for i in range(stopIter):
			# Compute data for each training pair
			for pair in pattPairs:
				outputMid = [ Decimal(0.0) for i in range(midLayerSize) ]
				outputOut = [ Decimal(0.0) for i in range(LayerSizes.OUT) ]
				deltaOut = None
				deltaMid = [ Decimal(0.0) for i in range(midLayerSize) ]
				errorOut = None
				errorMid = None
						
				# Compute outputs for middle layer
				for idx in range(midLayerSize):
					for i in range(LayerSizes.IN):
						outputMid[idx] = outputMid[idx] + (Decimal(pair.inputPatt[i]) * inMiddle[i][idx])

				# Compute outputs for output layer
				for idx in range(LayerSizes.OUT):
					for i in range(midLayerSize):
						outputOut[idx] = outputOut[idx] + (middleOut[i][idx] * outputMid[idx])

					outputOut[idx] = self.__sigFunc(outputOut[idx])

				# Compute error factors and errors
				deltaOut = [ (Decimal(pair.expectOut[idx]) - outputOut[idx]) for idx in range(LayerSizes.OUT) ]

				errorOut = [ (outputOut[idx] * (1 - outputOut[idx]) * deltaOut[idx]) for idx in range(LayerSizes.OUT) ]

				for i in range(midLayerSize):
					for idx in range(LayerSizes.OUT):
						deltaMid[i] = deltaMid[i] + (errorOut[idx] * middleOut[i][idx])

				errorMid = [ (outputMid[idx] * (1 - outputMid[idx]) * deltaMid[idx]) for idx in range(midLayerSize) ]

				# Update link weights
				for j in range(midLayerSize):
					for i in range(LayerSizes.IN):
						inMiddle[i][j] = inMiddle[i][j] + (learnRate * outputMid[j] * errorMid[j])

				for j in range(LayerSizes.OUT):
					for i in range(midLayerSize):
						middleOut[i][j] = middleOut[i][j] + (learnRate * outputOut[j] * errorOut[j])

		# Write data to a file
		with open(trainOutputFile, "w") as f:
			for line in inMiddle:
				for val in line:
					f.write("{!s} ".format(val))

				f.write("\n")

			f.write("\n")

			for line in middleOut:
				for val in line:
					f.write("{!s} ".format(val))

				f.write("\n")

			f.close()

	# Recognizes a pattern
	def recognize(self, netFilename, pattern):
		inMiddle = []
		middleOut = []

		numberRegexp = re.compile("([0-9]+\.[0-9]+)")
		idx = 0

		# Read the selected train file
		with open(netFilename, "r") as f:
			for line in f:
				if line.strip():
					valList = [ Decimal(item) for item in numberRegexp.findall(line) ]

					if idx < 48:
						inMiddle.append(valList)
					else:
						middleOut.append(valList)

				idx = idx + 1

			f.close()

		midLayerSize = len(middleOut)

		# Recognizes the given pattern
		outputMid = [ None for i in range(midLayerSize) ]
		outputOut = [ None for i in range(LayerSizes.OUT) ]

		# Compute outputs for middle layer
		for idx in range(midLayerSize):
			outputMid[idx] = sum(Decimal(pattern[i]) * inMiddle[i][idx] for i in range(LayerSizes.IN))
		
		# Compute outputs for output layer
		for idx in range(LayerSizes.OUT):
			outputOut[idx] = self.__sigFunc(sum(middleOut[i][idx] * outputMid[idx] for i in range(midLayerSize)))

		return outputOut
