#!/usr/bin/env python
import sys
from PySide import *
from plate_recog import *

class RecognizerWindow:
	def __init__(self):
		self.neuralNet = NeuralNet()

	def buildTrainScreen(self):
		trainScreen = QtGui.QWidget()
		layout = QtGui.QVBoxLayout()

		fileInput = QtGui.QWidget()
		fileInputLayout = QtGui.QHBoxLayout()
		self.txtFilePath = QtGui.QLineEdit()
		self.btnSelectFile = QtGui.QPushButton("Arquivo...")
		fileInputLayout.addWidget(self.txtFilePath)
		fileInputLayout.addWidget(self.btnSelectFile)
		fileInput.setLayout(fileInputLayout)
		layout.addWidget(fileInput)

		self.progressBar = QtGui.QProgressBar()
		layout.addWidget(self.progressBar)
		
		trainScreen.setLayout(layout)

		return trainScreen

	def buildMainScreen(self):
		mainScreen = QtGui.QHBoxLayout()

		mainScreen.addWidget(self.buildTrainScreen())

		return mainScreen

	def run(self):
		app = QtGui.QApplication(sys.argv)

		widget = QtGui.QWidget();
		widget.setLayout(self.buildMainScreen())
		widget.show()

		app.exec_()

if __name__ == "__main__":
	window = RecognizerWindow()
	window.run()
