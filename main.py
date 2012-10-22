#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from PySide import *
from plate_recog import *

# Main window
class RecognizerWindow(QtGui.QWidget):
	COL_QTY = 6
	LIN_QTY = 8

	def __init__(self):
		super(RecognizerWindow, self).__init__()
		self.neuralNet = NeuralNet()
		self.setLayout(self.buildMainScreen())
		self.trained = False
		self.answers = {
					"000000" : "0" , 
					"000001" : "1" , 
					"000010" : "2" , 
					"000011" : "3" , 
					"000100" : "4" , 
					"000101" : "5" , 
					"000110" : "6" , 
					"000111" : "7" , 
					"001000" : "8" , 
					"001001" : "9" , 
					"001010" : "A" , 
					"001011" : "B" , 
					"001100" : "C" , 
					"001101" : "D" , 
					"001110" : "E" , 
					"001111" : "F" , 
					"010000" : "G" , 
					"010001" : "H" , 
					"010010" : "I" , 
					"010011" : "J" ,
					"010100" : "K" , 
					"010101" : "L" , 
					"010110" : "M" , 
					"010111" : "N" , 
					"011000" : "O" , 
					"011001" : "P" , 
					"011010" : "Q" , 
					"011011" : "R" , 
					"011100" : "S" , 
					"011101" : "T" , 
					"011110" : "U" , 
					"011111" : "V" , 
					"100000" : "W" , 
					"100001" : "X" , 
					"100010" : "Y" , 
					"100011" : "Z" }

	def buildTrainScreen(self):
		trainScreen = QtGui.QGroupBox(self.trUtf8("Treinamento"))
		layout = QtGui.QVBoxLayout()

		form = QtGui.QWidget()
		formLayout = QtGui.QFormLayout()

		fileInput = QtGui.QWidget()
		fileInputLayout = QtGui.QHBoxLayout()
		self.txtFilePath = QtGui.QLineEdit()
		self.btnSelectFile = QtGui.QPushButton(self.trUtf8("Arquivo..."))
		self.btnSelectFile.clicked.connect(self.selectFilePatt)
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

		layout = QtGui.QVBoxLayout()

		form = QtGui.QWidget()
		formLayout = QtGui.QFormLayout()
	
		fileInput = QtGui.QWidget()
		fileInputLayout = QtGui.QHBoxLayout()
		self.txtRecogFilePath = QtGui.QLineEdit()
		self.btnRecogSelectFile = QtGui.QPushButton(self.trUtf8("Arquivo..."))
		self.btnRecogSelectFile.clicked.connect(self.selectFileTrain)
		fileInputLayout.addWidget(self.txtRecogFilePath)
		fileInputLayout.addWidget(self.btnRecogSelectFile)
		fileInput.setLayout(fileInputLayout)
		formLayout.addRow(self.trUtf8("Arquivo de treinamento: "), fileInput)

		form.setLayout(formLayout)
		layout.addWidget(form)

		self.btnLoadTrainFile = QtGui.QPushButton(self.trUtf8("Carregar treinamento selecionado"))
		self.btnLoadTrainFile.clicked.connect(self.loadTrain)
		layout.addWidget(self.btnLoadTrainFile)

		self.lblLoadedTrain = QtGui.QLabel(self.trUtf8("Não há treinamento carregado!"))
		self.lblLoadedTrain.setStyleSheet("QLabel { color: red; font-style: italic; }")
		layout.addWidget(self.lblLoadedTrain)

		self.buttonPadList = []
		buttonPadLayout = QtGui.QGridLayout()

		for j in range(RecognizerWindow.LIN_QTY):
			line = []
			for i in range(RecognizerWindow.COL_QTY):
				button = QtGui.QPushButton()
				button.setCheckable(True)
				button.setMaximumSize(30,30)
				button.setStyleSheet("QPushButton:checked { background-color: navy; }")
				line.append(button)
				buttonPadLayout.addWidget(button, j, i)
		
			self.buttonPadList.append(line)

		buttonPad = QtGui.QWidget()
		buttonPad.setLayout(buttonPadLayout)
		layout.addWidget(buttonPad)

		self.btnRecogPattern = QtGui.QPushButton(self.trUtf8("Reconhecer padrão"))
		self.btnRecogPattern.clicked.connect(self.recognize)
		layout.addWidget(self.btnRecogPattern)

		resultBox = QtGui.QWidget()
		resultLayout = QtGui.QVBoxLayout()
		self.lblResultText = QtGui.QLabel(self.trUtf8("Resultado"))
		resultLayout.addWidget(self.lblResultText)

		self.lblResult = QtGui.QLabel()
		self.lblResult.setStyleSheet("QLabel { font-size: 40px; color: orange; }")
		resultLayout.addWidget(self.lblResult)
		resultBox.setLayout(resultLayout)

		layout.addWidget(resultBox)

		recognizeScreen.setLayout(layout)

		return recognizeScreen

	def buildMainScreen(self):
		mainScreen = QtGui.QVBoxLayout()

		mainScreen.addWidget(self.buildTrainScreen())
		mainScreen.addWidget(self.buildRecognizeScreen())

		return mainScreen

	def selectFilePatt(self):
		filename = QtGui.QFileDialog.getOpenFileName(self,\
			self.trUtf8("Selecionar arquivo"), "",\
			self.trUtf8("Text files (*.txt)"))

		if filename:
			self.txtFilePath.setText(filename[0])
	
	def selectFileTrain(self):
		filename = QtGui.QFileDialog.getOpenFileName(self,\
			self.trUtf8("Selecionar arquivo"), "",\
			self.trUtf8("Train files (*.trn)"))

		if filename:
			self.txtRecogFilePath.setText(filename[0])
	
	def doTrain(self):
		numberIterations = self.txtNumIterations.value()
		self.progressBar.reset()
		self.progressBar.setRange(0, numberIterations - 1)
		self.neuralNet.train(self.txtFilePath.text(), self.txtNumNeurons.value(),\
			self.txtLearningRate.value(), numberIterations,	self.progressBar)

		self.trained = True
		self.updateLabelTrain()

	def doSaveTrain(self):
		filename = QtGui.QFileDialog.getSaveFileName(self,\
			self.trUtf8("Salvar treinamento"), "", \
			self.trUtf8("Train files (*.trn)"))

		netFileName = filename[0]
		if filename\
			and netFileName.find(".trn") == len(netFileName) - 4:
			netFileName = netFileName + ".trn"

		self.neuralNet.exportNet(netFileName)
	
	def loadTrain(self):
		filename = self.txtRecogFilePath.text()

		if filename:
			self.neuralNet.importNet(filename)

		self.trained = True
		self.updateLabelTrain()
	
	def recognize(self):
		if self.trained:
			pattern = StringIO.StringIO()

			for l in self.buttonPadList:
				for item in l:
					if item.isChecked():
						pattern.write("1")
					else:
						pattern.write("0")

			res = self.neuralNet.recognize(pattern.getvalue())
			pattern.close()

			self.lblResult.setText(self.answers.get(res))
	
	def updateLabelTrain(self):
		if self.trained:
			self.lblLoadedTrain.setText(self.trUtf8("Treinamento carregado!"))
			self.lblLoadedTrain.setStyleSheet(" QLabel { color: navy; font-weight: bold; }")

if __name__ == "__main__":
	app = QtGui.QApplication(sys.argv)

	r = RecognizerWindow()
	r.show()

	app.exec_()
	sys.exit()
