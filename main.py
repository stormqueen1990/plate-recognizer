#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from PySide import *
from plate_recog import *

class RecognizerWindow(QtGui.QWidget):
	def __init__(self):
		super(RecognizerWindow, self).__init__()
		self.neuralNet = NeuralNet()
		self.setLayout(self.buildMainScreen())

	def buildTrainScreen(self):
		trainScreen = QtGui.QGroupBox(self.trUtf8("Treinamento"))
		layout = QtGui.QVBoxLayout()

		form = QtGui.QWidget()
		formLayout = QtGui.QFormLayout()

		fileInput = QtGui.QWidget()
		fileInputLayout = QtGui.QHBoxLayout()
		self.txtFilePath = QtGui.QLineEdit()
		self.btnSelectFile = QtGui.QPushButton(self.trUtf8("Arquivo..."))
		self.btnSelectFile.clicked.connect(self.selectFile)
		fileInputLayout.addWidget(self.txtFilePath)
		fileInputLayout.addWidget(self.btnSelectFile)
		fileInput.setLayout(fileInputLayout)
		formLayout.addRow(self.trUtf8("Arquivo de padrões: "), fileInput)

		self.txtNumNeurons = QtGui.QSpinBox()
		self.txtNumNeurons.setValue(20)
		formLayout.addRow(self.trUtf8("Nº neurônios camada intermediária: "), self.txtNumNeurons)

		self.txtLearningRate = QtGui.QDoubleSpinBox()
		self.txtLearningRate.setValue(0.5)
		self.txtLearningRate.setRange(0.0, 1.0)
		formLayout.addRow(self.trUtf8("Taxa de aprendizagem: "), self.txtLearningRate)

		self.txtNumIterations = QtGui.QSpinBox()
		self.txtNumIterations.setRange(10,100000)
		self.txtNumIterations.setValue(500)
		formLayout.addRow(self.trUtf8("Número de iterações: "), self.txtNumIterations)

		form.setLayout(formLayout)
		layout.addWidget(form)


		buttonBar = QtGui.QWidget()
		buttonBarLayout = QtGui.QHBoxLayout()
		self.btnTrain = QtGui.QPushButton(self.trUtf8("Iniciar treinamento"))
		self.btnTrain.clicked.connect(self.doTrain)
		
		self.btnSaveTrain = QtGui.QPushButton(self.trUtf8("Salvar treinamento"))
		self.btnSaveTrain.clicked.connect(self.doSaveTrain)

		buttonBarLayout.addWidget(self.btnTrain)
		buttonBarLayout.addWidget(self.btnSaveTrain)
		buttonBar.setLayout(buttonBarLayout)

		layout.addWidget(buttonBar)

		self.progressBar = QtGui.QProgressBar()
		layout.addWidget(self.progressBar)
		
		trainScreen.setLayout(layout)

		return trainScreen
	
	def buildRecognizeScreen(self):
		recognizeScreen = QtGui.QGroupBox(self.trUtf8("Reconhecimento"))
	
		fileInput = QtGui.QWidget()
		fileInputLayout = QtGui.QHBoxLayout()
		self.txtRecogFilePath = QtGui.QLineEdit()
		self.btnRecogSelectFile = QtGui.QPushButton(self.trUtf8("Arquivo..."))
		self.btnRecogSelectFile.clicked.connect(self.selectFile)
		fileInputLayout.addWidget(self.txtRecogFilePath)
		fileInputLayout.addWidget(self.btnRecogSelectFile)
		fileInput.setLayout(fileInputLayout)
		formLayout.addRow(self.trUtf8("Arquivo de padrões: "), fileInput)



	def buildMainScreen(self):
		mainScreen = QtGui.QHBoxLayout()

		mainScreen.addWidget(self.buildTrainScreen())

		return mainScreen

	def selectFile(self):
		filename = QtGui.QFileDialog.getOpenFileName(self,\
			self.trUtf8("Selecionar arquivo"), "",\
			self.trUtf8("Text files (*.txt)"))
		self.txtFilePath.setText(filename[0])
	
	def doTrain(self):
		numberIterations = self.txtNumIterations.value()
		self.progressBar.reset()
		self.progressBar.setRange(0, numberIterations - 1)
		self.neuralNet.train(self.txtFilePath.text(), self.txtNumNeurons.value(),\
			self.txtLearningRate.value(), numberIterations,	self.progressBar)

	def doSaveTrain(self):
		filename = QtGui.QFileDialog.getSaveFileName(self,\
			self.trUtf8("Salvar treinamento"), "", \
			self.trUtf8("Train files (*trn)"))

		netFileName = filename[0]
		if netFileName.find(".trn") == len(netFileName) - 4:
			netFileName = netFileName + ".trn"

		self.neuralNet.exportNet(netFileName)

if __name__ == "__main__":
	app = QtGui.QApplication(sys.argv)

	r = RecognizerWindow()
	r.show()

	app.exec_()
	sys.exit()
