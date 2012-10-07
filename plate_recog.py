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

# Read and return all pattern pairs from file
def readPatternFile(patternFileName):
	# List with all pairs
	pattPairs = []

	# Creates the regexp to search for patterns
	regexpPatt = re.compile("([01]+)")

	# Searches the file for patterns
	fPatt = open(patternFileName, "r")

	for line in fPatt:
		it = regexpPatt.findall(line)

		# Does line contain a pair?
		if len(it) < 2:
			raise ValueError("all lines must contain a pair")

		p = PattPair(it[0], it[1])
		pattPairs.append(p)
	
	fPatt.close()

	return pattPairs

# Plate recognizement kernel
class PlateRecognizer:
	# Sigmoid function
	def __sigFunc(self, sumVal):
		return 1.0/(1.0+math.exp(-sumVal))

	# Trains a randomly constructed neural net
	def train(self, pattFile, trainOutputFile, midLayerSize, learnRate, stopIter):
		random.seed()
		# Constructs the initial training matrix
		inMiddle = [ [ float(random.randint(0,999))/float(1000) for i in range(midLayerSize) ] for i in range(LayerSizes.IN) ]
		middleOut = [ [ float(random.randint(0,999))/float(1000) for i in range(LayerSizes.OUT) ] for i in range(midLayerSize) ]

		# List with all pairs
		pattPairs = readPatternFile(pattFile)
	
		# Compute data for each training pair
		for pair in pattPairs:

			# Train pattern for x iterations
			for i in range(stopIter):
				outputMid = [ 0.0 for i in range(midLayerSize) ]
				outputOut = [ 0.0 for i in range(LayerSizes.OUT) ]
				deltaOut = None
				deltaMid = None
				errorOut = None
				errorMid = None
						
				# Compute outputs for middle layer
				for idx in range(midLayerSize):
					for i in range(LayerSizes.IN):
						outputMid[idx] = outputMid[idx] + (float(pair.inputPatt[i]) * inMiddle[i][idx])	
					
					outputMid[idx] = self.__sigFunc(outputMid[idx])

				# Compute outputs for output layer
				for idx in range(LayerSizes.OUT):
					for i in range(midLayerSize):
						outputOut[idx] = outputOut[idx] + (middleOut[i][idx] * outputMid[idx])

					outputOut[idx] = self.__sigFunc(outputOut[idx])

				# Compute error factors and errors
				deltaOut = [ (float(pair.expectOut[idx]) - outputOut[idx]) for idx in range(LayerSizes.OUT) ]

				errorOut = [ (outputOut[idx] * (1 - outputOut[idx]) * deltaOut[idx]) for idx in range(LayerSizes.OUT) ]

				deltaMid = [ sum(errorOut[idx] * middleOut[i][idx] for idx in range(LayerSizes.OUT)) for i in range(midLayerSize) ] 

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
	def recognize(self, netFilename, patternFile):
		inMiddle = []
		middleOut = []
		patterns = readPatternFile(patternFile)

		numberRegexp = re.compile("([0-9]+\.[0-9]+)")
		idx = 0

		# Read the selected train file
		with open(netFilename, "r") as f:
			for line in f:
				if line.strip():
					valList = [ float(item) for item in numberRegexp.findall(line) ]

					if idx < 48:
						inMiddle.append(valList)
					else:
						middleOut.append(valList)

				idx = idx + 1

			f.close()

		midLayerSize = len(middleOut)

		for pattern in patterns:
			# Recognizes the given pattern
			outputMid = [ None for i in range(midLayerSize) ]
			outputOut = [ None for i in range(LayerSizes.OUT) ]

			# Compute outputs for middle layer
			for idx in range(midLayerSize):
				outputMid[idx] = sum(float(pattern.inputPatt[i]) * inMiddle[i][idx] for i in range(LayerSizes.IN))
		
			# Compute outputs for output layer
			for idx in range(LayerSizes.OUT):
				outputOut[idx] = self.__sigFunc(sum(middleOut[i][idx] * self.__sigFunc(outputMid[idx]) for i in range(midLayerSize)))

			print outputOut
