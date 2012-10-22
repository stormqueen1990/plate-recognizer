#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import re
import math
import StringIO
from PySide import *

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

# Neuron representation
class Neuron:
	def __init__(self):
		self.weights = None

	def initializeWeights(self, size):
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

# Input layer neuron specialization
class InputNeuron(Neuron):
	def __init__(self):
		Neuron.__init__(self)
		self.inputValue = None

	def calculateOut(self):
		self.output = self.inputValue

	def calculateErrorValue(self):
		raise NotImplemented("not implemented for this class")

	def updateWeights(self, inputLayer, learnRate):
		raise NotImplemented("not implemented for this class")
	
	def setInputs(self, inputValue):
		self.inputValue = inputValue

# Middle layer neuron specialization
class MiddleNeuron(Neuron):
	def __init__(self, initWeights):
		Neuron.__init__(self)

		if initWeights:
			self.initializeWeights(LayerSizes.IN)

	def calculateErrorFactor(self, outputLayer):
		self.errorFactor = sum(outputLayer[i].errorValue * self.weights[i] for i in range(LayerSizes.OUT))

	def setInputs(self, inputs):
		self.inputs = inputs

# Output layer neuron specialization
class OutputNeuron(Neuron):
	def __init__(self, middleLayerSize = None):
		Neuron.__init__(self)
		self.expectedOutput = None

		if middleLayerSize != None:
			self.initializeWeights(middleLayerSize)

	def calculateErrorFactor(self):
		self.errorFactor = self.expectedOutput - self.output
	
	def setInputs(self, inputs):
		self.inputs = inputs

class WrongFormatException(Exception):
	def __init__(self, message):
		self.message = message
	
	def __str__(self):
		return self.message

class UnexpectedSizeException(Exception):
	def __init__(self, message):
		self.message = message
	
	def __str__(self):
		return self.message
	
class NeuralNet:
	def __init__(self):
		self.inputLayer = None
		self.middleLayer = None
		self.outputLayer = None
	
	# Read and return all pattern pairs from file
	def readPatternFile(self, patternFileName):
		# List with all pairs
		pattPairs = []

		# Creates the regexp to search for patterns
		regexpPatt = re.compile("([01]+)")

		# Searches the file for patterns
		fPatt = open(patternFileName, "r")

		for line in fPatt:
			it = regexpPatt.findall(line)

			# Does line contain three entries?
			if len(it) < 2:
				raise WrongFormatException(u"todas as linhas do arquivo " + patternFileName + u" deveriam conter um par separado por espaço, seguido do valor representado pelo par")

			if len(it[0]) != LayerSizes.IN:
				raise UnexpectedSizeException(u"o padrão de entrada deve conter " + `LayerSizes.IN` + " caracteres")

			if len(it[1]) != LayerSizes.OUT:
				raise UnexpectedSizeException(u"a saída esperada deve conter " + `LayerSizes.OUT` + " caracteres")

			p = PattPair(it[0], it[1])
			pattPairs.append(p)
	
		fPatt.close()

		return pattPairs

	# Trains the network
	def train(self, patternFileName, middleLayerSize, learnRate, iterNumber, progressBar):
		# Patterns
		patternPairs = self.readPatternFile(patternFileName)
		# Layers
		inputLayer = [ InputNeuron() for i in range(LayerSizes.IN) ]
		middleLayer = [ MiddleNeuron(True) for i in range(middleLayerSize) ]
		outputLayer = [ OutputNeuron(middleLayerSize) for i in range(LayerSizes.OUT) ]
		
		# Repeats for the selected number of iterations
		for i in range(iterNumber):
			for pair in patternPairs:
				idx = 0

				# Feeds the input layer
				for bit in pair.inputPatt:
					inputLayer[idx].inputValue = float(bit)
					idx = idx + 1

				idx = 0
				for bit in pair.expectOut:
					outputLayer[idx].expectedOutput = float(bit)
					idx = idx + 1
				
				# Feeds the middle layer and calculates outputs
				middleLayerInputs = []
				for neuron in inputLayer:
					neuron.calculateOut()
					middleLayerInputs.append(neuron.output)

				# Feeds the output layer
				outputLayerInputs = []
				for neuron in middleLayer:
					neuron.setInputs(middleLayerInputs)
					neuron.calculateOut()
					outputLayerInputs.append(neuron.output)

				# Backpropagation
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

			# Updates the progress bar
			progressBar.setValue(i)

		# Storing layers
		self.inputLayer = inputLayer
		self.middleLayer = middleLayer
		self.outputLayer = outputLayer
	
	# Tries to recognize a given pattern
	def recognize(self, pattern):
		idx = 0
		# Feeds input layer
		for bit in pattern:
			self.inputLayer[idx].inputValue = float(bit)
			idx = idx + 1
		
		# Process values to output
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

		# Prepare a string with the obtained answer and returns
		return self.__prepareAnswer()
		
	# Prepares a binary string from the neural net output
	def __prepareAnswer(self):
		answerBuilder = StringIO.StringIO()
		
		for neuron in self.outputLayer:
			if neuron.output >= 0.5:
				answerBuilder.write("1")
			else:
				answerBuilder.write("0")

		answerBuilder.flush()
		answer = answerBuilder.getvalue()
		answerBuilder.close()
	
		return answer
	
	# Exports the neural network to a XML file
	def exportNet(self, neuralNetFilename):
		neuralNetFile = QtCore.QFile(neuralNetFilename)
		if neuralNetFile.open(QtCore.QIODevice.WriteOnly):
			netFileWriter = QtCore.QXmlStreamWriter(neuralNetFile)
			netFileWriter.setAutoFormatting(True)
			netFileWriter.setAutoFormattingIndent(-1)
			netFileWriter.writeStartDocument()

			netFileWriter.writeStartElement(u"neuralNet")

			self.__exportLayer(netFileWriter, u"middleNode", self.middleLayer)
			self.__exportLayer(netFileWriter, u"outputNode", self.outputLayer)

			netFileWriter.writeEndElement() # neuralNet

			netFileWriter.writeEndDocument()
			neuralNetFile.close()
	
	# Exports each layer to file
	def __exportLayer(self, netFileWriter, elementName, layer):
		idx = 0
		for neuron in layer:
			netFileWriter.writeStartElement(elementName)
			netFileWriter.writeAttribute(u"id", unicode(idx))
			idx = idx + 1
			netFileWriter.writeStartElement(u"weights")

			for weight in neuron.weights:
				netFileWriter.writeStartElement(u"weight")
				netFileWriter.writeAttribute(u"value", unicode(weight))
				netFileWriter.writeEndElement() # weight

			netFileWriter.writeEndElement() # weights
			netFileWriter.writeEndElement() # elementName
	
	# Feeds the neural net with data from given file
	def importNet(self, neuralNetFilename):
		neuralNetFile = QtCore.QFile(neuralNetFilename)
		if neuralNetFile.open(QtCore.QIODevice.ReadOnly):
			netFileReader = QtCore.QXmlStreamReader(neuralNetFile)

			self.inputLayer = [ InputNeuron() for i in range(LayerSizes.IN) ]
			self.middleLayer = []
			self.outputLayer = []

			weights = []
			while not netFileReader.atEnd():
				token = netFileReader.readNext()
				if token == QtCore.QXmlStreamReader.StartElement:
					if netFileReader.name() == u"weight":
						weight = netFileReader.attributes().value(u"value")
						weights.append(float(weight))
				elif token == QtCore.QXmlStreamReader.EndElement:
					if netFileReader.name() == u"middleNode":
						neuron = MiddleNeuron(False)
						neuron.weights = weights
						self.middleLayer.append(neuron)
						weights = []
					elif netFileReader.name() == u"outputNode":
						neuron = OutputNeuron()
						neuron.weights = weights
						self.outputLayer.append(neuron)
						weights = []

			neuralNetFile.close()
