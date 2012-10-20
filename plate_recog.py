#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import re
import math
import StringIO

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

# Pattern pairs class
class PattPair:
	def __init__(self, inputPatt, expectOut):
		self.inputPatt = inputPatt
		self.expectOut = expectOut

class Neuron:		
	def __init__(self, size):
		if size > 0:
			self.weights = [ random.random() for i in range(size) ]

	def __sigmoidFunction(self, value):
		return 1.0 / (1.0 + math.exp(-value))
	
	def calculateOut(self):
		sumValue = sum(self.weights[i] * self.inputs[i] for i in range(len(self.weights)))
		self.output = self.__sigmoidFunction(sumValue)
	
	def calculateErrorValue(self):
		self.errorValue = self.output * (1.0 - self.output) * self.errorFactor
	
	def updateWeights(self, inputLayer, learnRate):
		for i in range(len(self.weights)):
			self.weights[i] = self.weights[i] + (learnRate * inputLayer[i].output * self.errorValue)

class InputNeuron(Neuron):
	def __init__(self):
		Neuron.__init__(self, 0)

	def calculateOut(self):
		self.output = self.inputValue

	def calculateErrorValue(self):
		raise NotImplemented("not implemented for this class")

	def updateWeights(self, inputLayer, learnRate):
		raise NotImplemented("not implemented for this class")
	
	def setInputs(self, inputValue):
		self.inputValue = inputValue

class MiddleNeuron(Neuron):
	def __init__(self, middleLayerSize):
		Neuron.__init__(self, middleLayerSize)

	def calculateErrorFactor(self, outputLayer):
		self.errorFactor = sum(outputLayer[i].errorValue * self.weights[i] for i in range(LayerSizes.OUT))

	def setInputs(self, inputs):
		self.inputs = inputs
	
class OutputNeuron(Neuron):
	def __init__(self):
		Neuron.__init__(self, LayerSizes.OUT)

	def calculateErrorFactor(self):
		self.errorFactor = self.expectedOutput - self.output
	
	def setInputs(self, inputs):
		self.inputs = inputs
	
	def setExpectedOutput(self, expectedOutput):
		self.expectedOutput = expectedOutput
	
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

class NeuralNet:
	def __init__(self):
		self.inputLayer = None
		self.middleLayer = None
		self.outputLayer = None

	def train(self, patternFileName, outputFileName, middleLayerSize, learnRate, iterNumber):
		patternPairs = readPatternFile(patternFileName)
		inputLayer = [ InputNeuron() for i in range(LayerSizes.IN) ]
		middleLayer = [ MiddleNeuron(middleLayerSize) for i in range(middleLayerSize) ]
		outputLayer = [ OutputNeuron() for i in range(LayerSizes.OUT) ]
		
		for i in range(iterNumber):
			for pair in patternPairs:
				idx = 0
				for bit in pair.inputPatt:
					inputLayer[idx].inputValue = float(bit)
					idx = idx + 1

				idx = 0
				for bit in pair.expectOut:
					outputLayer[idx].setExpectedOutput(float(bit))
					idx = idx + 1
				
				middleLayerInputs = []
				for neuron in inputLayer:
					neuron.calculateOut()
					middleLayerInputs.append(neuron.output)

				outputLayerInputs = []
				for neuron in middleLayer:
					neuron.setInputs(middleLayerInputs)
					neuron.calculateOut()
					outputLayerInputs.append(neuron.output)

				for neuron in outputLayer:
					neuron.setInputs(outputLayerInputs)
					neuron.calculateOut()
					neuron.calculateErrorFactor()
					neuron.calculateErrorValue()

				for neuron in middleLayer:
					neuron.calculateErrorFactor(outputLayer)
					neuron.calculateErrorValue()
					neuron.updateWeights(inputLayer, learnRate)

				for neuron in outputLayer:
					neuron.updateWeights(middleLayer, learnRate)

		self.inputLayer = inputLayer
		self.middleLayer = middleLayer
		self.outputLayer = outputLayer
	
	def recognize(self, pattern):
		idx = 0
		for bit in pattern:
			self.inputLayer[idx].inputValue = float(bit)
			idx = idx + 1
		
		middleLayerInputs = []
		for neuron in self.inputLayer:
			neuron.calculateOut()
			middleLayerInputs.append(neuron.output)

		outputLayerInputs = []
		for neuron in self.middleLayer:
			neuron.setInputs(middleLayerInputs)
			neuron.calculateOut()
			outputLayerInputs.append(neuron.output)

		for neuron in self.outputLayer:
			neuron.setInputs(outputLayerInputs)
			neuron.calculateOut()

		return self.__prepareAnswer(self.outputLayer)
		
	def __prepareAnswer(self):
		answerBuilder = StringIO.StringIO()
		
		for neuron in self.outputLayer:
			answerBuilder.write(1 if neuron.output >= 0.5 else 0)

		answer = answerBuilder.getvalue()
		answerBuilder.close()

		return answer
