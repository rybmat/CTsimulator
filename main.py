import CTsim
from skimage import data_dir
import sys
from PyQt4 import QtGui, QtCore
import numpy as np
from PIL import Image

def spinBox(parent, prefix, minSizeX, minSizeY, minVal, maxVal, defVal, posX, posY):
	sb = QtGui.QSpinBox(parent)
	sb.setPrefix(prefix)
	sb.setMinimumSize(minSizeX, minSizeY)
	sb.setRange(minVal, maxVal)
	sb.setValue(defVal)
	sb.move(posX, posY)

	return sb


class CTSimGui(QtGui.QMainWindow):

	def __init__(self):
		super(CTSimGui, self).__init__()
		
		self.fname = ""
		self.fft_filter_type = "ramp"

		self.initUI()
		
		
	def initUI(self):      

		#layout
		leftFrame = QtGui.QFrame(self)
		leftFrame.setFrameShape(QtGui.QFrame.StyledPanel)
		
		rightFrame = QtGui.QFrame(self)
		rightFrame.setFrameShape(QtGui.QFrame.StyledPanel)

		splitter1 = QtGui.QSplitter(QtCore.Qt.Horizontal)
		splitter1.addWidget(leftFrame)
		splitter1.addWidget(rightFrame)

		self.setCentralWidget(splitter1)


		#label with fname
		self.fname_label = QtGui.QLabel("Selected File: ", leftFrame)
		self.fname_label.setMinimumSize(300,10)
		self.fname_label.move(10,5)

		#rotation angle
		self.rot_angle_sb = spinBox(leftFrame, "Rotation Angle [deg]:    ", 200, 10, 1, 360, 180, 5,30)

		#rotation step
		self.rot_step_sb = spinBox(leftFrame, "Rotation Step [deg]:      ", 200, 10, 1, 90, 30, 5,65)

		#detectors number
		self.det_num_sb = spinBox(leftFrame, "Detectors Number:       ", 200, 10, 1, 999999, 500, 5, 100)

		#detector size
		self.det_size_sb = spinBox(leftFrame, "Detector Size [px]:        ", 200, 10, 1, 100, 2, 5, 135)

		#emmiter distance
		self.emm_dist_sb = spinBox(leftFrame, "Emmiter Distance [px]: ", 200, 10, 1, 999999, 500, 5, 170)

		#detectors distance
		self.det_dist_sb = spinBox(leftFrame, "Detector Distance [px]: ", 200, 10, 1, 999999, 500, 5, 205)

		#fft/normal filter
		self.fft_cb = QtGui.QCheckBox("Use FFT filter", leftFrame)
		self.fft_cb.move(5, 240)
		self.fft_cb.stateChanged.connect(self.fft_cbStateChanged)

		#choose fft filter
		self.fft_filter_cb = QtGui.QComboBox(leftFrame)
		self.fft_filter_cb.move(10,265)
		self.fft_filter_cb.setEnabled(False)
		self.fft_filter_cb.addItem("ramp")
		self.fft_filter_cb.addItem("shepp-logan")
		self.fft_filter_cb.addItem("cosine")
		self.fft_filter_cb.addItem("hamming")
		self.fft_filter_cb.addItem("hann")
		self.fft_filter_cb.activated[str].connect(self.comboSelect)

		#rgb check box
		self.raysMode_cb = QtGui.QCheckBox("1 generation CT", leftFrame)
		self.raysMode_cb.move(5, 300)

		#image label
		self.pix_label = QtGui.QLabel(rightFrame)


		self.statusBar()

		#open file action
		openFile = QtGui.QAction(QtGui.QIcon('rsc/icons/openfile.png'), 'Select Image', self)
		openFile.setShortcut('Ctrl+O')
		openFile.setStatusTip('Select Image')
		openFile.triggered.connect(self.openFileDialog)

		#run algorithm action
		run = QtGui.QAction(QtGui.QIcon('rsc/icons/run.png'), 'Run', self)
		run.setShortcut('Ctrl+R')
		run.setStatusTip('Run')
		run.triggered.connect(self.runAlgorithm)

		#toolbar
		self.toolbar = self.addToolBar('Run')
		self.toolbar.addAction(openFile)
		self.toolbar.addAction(run)

		#menu
		menubar = self.menuBar()
		fileMenu = menubar.addMenu('&File')
		fileMenu.addAction(openFile) 
		fileMenu.addAction(run)      
		

		#window
		self.setGeometry(300, 300, 600, 500)
		self.setWindowTitle('CT Simulator')
		self.show()
		
	def openFileDialog(self):
		self.fname = QtGui.QFileDialog.getOpenFileName(self, 'Open file', '~/', "*.png")
		self.onOpenFile()


	def onOpenFile(self):
		self.fname_label.setText("Selected File: " + self.fname.split("/")[-1])
		self.fname_label.adjustSize()

		self.pix_label.setPixmap(QtGui.QPixmap(self.fname))
		self.pix_label.adjustSize()



	def comboSelect(self, text):
		self.fft_filter_type = text
		 

	def runAlgorithm(self):
		if (self.rot_step_sb.value() > self.rot_angle_sb.value()):
			print "rotation step have to be smaller or equal than rotation angle"
			QtGui.QMessageBox.information(self, 'Message', "Rotation step have to be smaller or equal than rotation angle", QtGui.QMessageBox.Ok)
			return

		print "fname", self.fname
		print "rotation angle: ", self.rot_angle_sb.value()
		print "rotation step: ", self.rot_step_sb.value()
		print "detectors number: ", self.det_num_sb.value()
		print "detector size: ", self.det_size_sb.value()
		print "emmter distance: ", self.emm_dist_sb.value()
		print "detectors distance: ", self.det_dist_sb.value()
		print "use fft filter: ", self.fft_cb.isChecked()
		print "fft filter type: ", self.fft_filter_type
		print "Rays Mode: ", self.raysMode_cb.isChecked()
		print "============"


		if self.fname == "":
			file_path =  data_dir + "/phantom.png"
			#file_path =  "RGB.png"
		else:
			file_path = self.fname

		a = CTsim.CTsimRadon(image_path=str(file_path), angle=self.rot_angle_sb.value(), step=self.rot_step_sb.value(), detNum=self.det_num_sb.value(), detSize=self.det_size_sb.value(), emmDist=self.emm_dist_sb.value(), detDist=self.det_dist_sb.value(), fft=self.fft_cb.isChecked(), filter=self.fft_filter_type, mode=self.raysMode_cb.isChecked())
		a.run(show = False)
		img = a.getImage()
		
		nimage = QtGui.QImage(img.data,img.shape[0],img.shape[1],QtGui.QImage.Format_Indexed8)
		nimage.ndarray = img
		for i in range(256):
			nimage.setColor(i, QtGui.QColor(i,i,i).rgb())

		
		#qimg = QtGui.QImage(img, img.shape[1], img.shape[0], QtGui.QImage.Format_RGB16)
		self.pix_label.setPixmap(QtGui.QPixmap.fromImage(nimage))	
		self.pix_label.adjustSize()	

	def fft_cbStateChanged(self, a):
		self.fft_filter_cb.setEnabled(a)


def main():
	
	app = QtGui.QApplication(sys.argv)
	ex = CTSimGui()
	sys.exit(app.exec_())

if __name__ == '__main__':
	main()   
