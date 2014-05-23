import CTsim
from skimage import data_dir
from skimage.transform import resize
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

class Frame(QtGui.QFrame):
	def __init__(self, parent):
		super(Frame, self).__init__(parent)
		self.widget = ""

	def add(self, widget):
		self.widget = widget

	def resizeEvent (self, resEv):
		super(Frame, self).resizeEvent(resEv)
		if self.widget:
			self.widget.setFixedWidth(self.width())
			self.widget.setFixedHeight(self.height())



class CTSimGui(QtGui.QMainWindow):

	def __init__(self):
		super(CTSimGui, self).__init__()
		
		self.fname = ""
		self.fft_filter_type = "ramp"
		self.firstGen = False

		self.initUI()
		
		
	def initUI(self):      

		#layout
		leftFrame = QtGui.QFrame(self)
		leftFrame.setFrameShape(QtGui.QFrame.StyledPanel)
		leftFrame.setFixedWidth(300)
		
		self.rightFrame = Frame(self)
		self.rightFrame.setFrameShape(QtGui.QFrame.StyledPanel)


		splitter1 = QtGui.QSplitter(QtCore.Qt.Horizontal)
		splitter1.addWidget(leftFrame)
		splitter1.addWidget(self.rightFrame)


		self.setCentralWidget(splitter1)


		#label with fname
		self.fname_label = QtGui.QLabel("Selected File: ", leftFrame)
		self.fname_label.setMinimumSize(300,10)
		self.fname_label.move(5,500)

		#generation select
		self.generation_sel = QtGui.QComboBox(leftFrame)
		self.generation_sel.move(5, 20)
		self.generation_sel.addItem("III generation CT")
		self.generation_sel.addItem("I generation CT")
		self.generation_sel.activated[str].connect(self.generationSelect)


		#rotation angle
		self.rot_angle_sb = spinBox(leftFrame, "Rotation Angle [deg]:    ", 200, 10, 1, 360, 180, 5,65)

		#rotation step
		self.rot_step_sb = spinBox(leftFrame, "Rotation Step [deg]:      ", 200, 10, 1, 90, 30, 5,100)

		#detectors number
		self.det_num_sb = spinBox(leftFrame, "Detectors Number:       ", 200, 10, 1, 999999, 500, 5, 135)

		#detector size
		self.det_size_sb = spinBox(leftFrame, "Detector Size [px]:        ", 200, 10, 1, 100, 2, 5, 170)

		#emmiter distance
		self.emm_dist_sb = spinBox(leftFrame, "Emmiter Distance [px]: ", 200, 10, 1, 999999, 500, 5, 205)

		#detectors distance
		self.det_dist_sb = spinBox(leftFrame, "Detector Distance [px]: ", 200, 10, 1, 999999, 500, 5, 240)

		#fft/normal filter
		self.fft_cb = QtGui.QCheckBox("Use FFT filter", leftFrame)
		self.fft_cb.move(5, 280)
		self.fft_cb.stateChanged.connect(self.fft_cbStateChanged)

		#choose fft filter
		self.fft_filter_cb = QtGui.QComboBox(leftFrame)
		self.fft_filter_cb.move(10,300)
		self.fft_filter_cb.setEnabled(False)
		self.fft_filter_cb.addItem("ramp")
		self.fft_filter_cb.addItem("shepp-logan")
		self.fft_filter_cb.addItem("cosine")
		self.fft_filter_cb.addItem("hamming")
		self.fft_filter_cb.addItem("hann")
		self.fft_filter_cb.activated[str].connect(self.comboSelect)

		#Brightness slider
		self.upCut_lbl = QtGui.QLabel(leftFrame)
		self.upCut_lbl.move(10,360)
		self.upCut_lbl.setText("Reconstructed image up cut")
		self.upCut_lbl.adjustSize()
		self.upCut_lbl.setEnabled(False)
		self.upCut_sl = QtGui.QSlider(QtCore.Qt.Horizontal, leftFrame)
		self.upCut_sl.setGeometry(10, 375, 200, 30)
		self.upCut_sl.valueChanged[int].connect(self.onChangeSlider)
		self.upCut_sl.setEnabled(False)
		self.upCut_sl.setValue(0)

		self.downCut_lbl = QtGui.QLabel(leftFrame)
		self.downCut_lbl.move(10,410)
		self.downCut_lbl.setText("Reconstructed image down cut")
		self.downCut_lbl.adjustSize()
		self.downCut_lbl.setEnabled(False)
		self.downCut_sl = QtGui.QSlider(QtCore.Qt.Horizontal, leftFrame)
		self.downCut_sl.setGeometry(10, 425, 200, 30)
		self.downCut_sl.valueChanged[int].connect(self.onChangeSlider)
		self.downCut_sl.setEnabled(False)
		self.downCut_sl.setValue(100)


		

		#image label
		self.pix_label = QtGui.QLabel()

		self.scroll = QtGui.QScrollArea(self.rightFrame)
		self.scroll.setWidget(self.pix_label)
		self.scroll.setWidgetResizable(True)
		self.scroll.setFixedHeight(self.rightFrame.height())
		self.scroll.setFixedWidth(self.rightFrame.width())
		self.rightFrame.add(self.scroll)
		

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
		self.setGeometry(100, 100, 1200, 700)
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

	def onChangeSlider(self):
		try:
			self.pix_label.setPixmap(QtGui.QPixmap.fromImage(self.prepareImage((self.upCut_sl.value() + 1) / 100.0, (self.downCut_sl.value() + 1) / 100.0)))	
			self.pix_label.adjustSize()
		except:
			pass


	def comboSelect(self, text):
		self.fft_filter_type = text

	def generationSelect(self, text):
		if text == "III generation CT":
			self.firstGen = False
		else:
			self.firstGen = True

		self.det_num_sb.setVisible(not self.firstGen)
		self.det_size_sb.setVisible(not self.firstGen)
		self.emm_dist_sb.setVisible(not self.firstGen)
		self.det_dist_sb.setVisible(not self.firstGen)
		 

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
		print "Rays Mode: ", self.firstGen
		print "============"


		if self.fname == "":
			file_path =  data_dir + "/phantom.png"
		else:
			file_path = self.fname

		self.a = CTsim.CTsimRadon(image_path=str(file_path), angle=self.rot_angle_sb.value(), step=self.rot_step_sb.value(), detNum=self.det_num_sb.value(), detSize=self.det_size_sb.value(), emmDist=self.emm_dist_sb.value(), detDist=self.det_dist_sb.value(), fft=self.fft_cb.isChecked(), filter=self.fft_filter_type, firstGen=self.firstGen)
		self.a.run(show = False)

		
		self.pix_label.setPixmap(QtGui.QPixmap.fromImage(self.prepareImage((self.upCut_sl.value() + 1) / 100.0, (self.downCut_sl.value() + 1) / 100.0)))	
		self.pix_label.adjustSize()	
		self.upCut_sl.setEnabled(True)
		self.upCut_lbl.setEnabled(True)
		self.downCut_sl.setEnabled(True)
		self.downCut_lbl.setEnabled(True)


	def fft_cbStateChanged(self, a):
		self.fft_filter_cb.setEnabled(a)

	def prepareImage(self, upCut=1.0, downCut=0.0):
		image = self.a.getImage()
		sinogram = self.a.getSinogram()
		reconstruction = self.a.getReconstruction()

		reconstruction_cutted = reconstruction
		for x in range(reconstruction_cutted.shape[0]):
			for y in range(reconstruction_cutted.shape[1]):
				if(reconstruction_cutted[x,y] < upCut):
					reconstruction_cutted[x,y] = upCut
				if(reconstruction_cutted[x,y] > downCut):
					reconstruction_cutted[x,y] = downCut
		reconstruction_cutted = self.a.normalize_array(reconstruction_cutted)
		
		
		result = np.zeros((image.shape[0]*2, image.shape[1]*2))
		result[:image.shape[0] , :image.shape[1]] = image
		result[image.shape[0]: , :image.shape[1]] = reconstruction_cutted
		result[:image.shape[0] , image.shape[1]:] = resize(sinogram, (image.shape[0], image.shape[1]))
		result[image.shape[0]: , image.shape[1]:] = self.a.normalize_array(reconstruction_cutted - image)
		
		result = result*255
		result = np.require(result, np.uint8, 'C')
		

		nimage = QtGui.QImage(result.data, result.shape[0], result.shape[1], QtGui.QImage.Format_Indexed8)
		nimage.ndarray = result
		for i in range(256):
			nimage.setColor(i, QtGui.QColor(i,i,i).rgb())

		return nimage


def main():
	
	app = QtGui.QApplication(sys.argv)
	ex = CTSimGui()
	sys.exit(app.exec_())

if __name__ == '__main__':
	main()   
