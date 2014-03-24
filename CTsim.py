import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage import data_dir
from skimage.transform import radon, rescale
from skimage.transform import iradon

class CTsimRadon:

	def __init__(self, image_path, angle):
		self.__image = imread(image_path, as_grey=True)
		self.__image = rescale(self.__image, scale=0.4)
		self.__angle = angle
		self.__theta = np.linspace(0., angle, max(self.__image.shape), endpoint=True)
		plt.figure(figsize=(10, 10))

	def __acquisition(self):
		self.__sinogram = radon(self.__image, theta=self.__theta, circle=True)

	def __reconstruction(self):
		self.__reconstruction_fbp = iradon(self.__sinogram, theta=self.__theta, circle=True)
		self.__error = self.__reconstruction_fbp - self.__image
		print('FBP rms reconstruction error: %.3g' % np.sqrt(np.mean(self.__error**2)))


	def __showSinogram(self, show=False):
		plt.subplot(221)
		plt.title("Original")
		plt.imshow(self.__image, cmap=plt.cm.Greys_r)

		plt.subplot(222)
		plt.title("Radon transform\n(Sinogram)")
		plt.xlabel("Projection angle (deg)")
		plt.ylabel("Projection position (pixels)")
		plt.imshow(self.__sinogram, cmap=plt.cm.Greys_r,
		           extent=(0, self.__angle, 0, self.__sinogram.shape[0]), aspect='auto')

		plt.subplots_adjust(hspace=0.4, wspace=0.5)
		if show:
			plt.show()

	def __showReconstruction(self, show=False):
		self.__imkwargs = dict(vmin=-0.2, vmax=0.2)
		
		plt.subplot(223)
		plt.title("Reconstruction\nFiltered back projection")
		plt.imshow(self.__reconstruction_fbp, cmap=plt.cm.Greys_r)
		
		plt.subplot(224)
		plt.title("Reconstruction error\nFiltered back projection")
		plt.imshow(self.__reconstruction_fbp - self.__image, cmap=plt.cm.Greys_r, **self.__imkwargs)
		
		if show:
			plt.show()

	def run(self):
		self.__acquisition()
		self.__showSinogram()
		self.__reconstruction()
		self.__showReconstruction()
		plt.show()

