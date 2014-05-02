from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage import data_dir
from skimage.transform import rescale, warp
#from skimage.transform import radon, iradon


from scipy.fftpack import fftshift, fft, ifft
from skimage.transform._warps_cy import _warp_fast
import time

class CTsimRadon:

	def __init__(self, image_path, angle, step, detNum, detSize, emmDist=500, detDist=500):
		self.__detectorsDistance = detDist
		self.__emmiterDistance = emmDist
		self.__detNum = detNum
		self.__detSize = detSize
		self.__image = imread(image_path, as_grey=True)
		self.__image = rescale(self.__image, scale=0.4)
		self.__angle = angle
		#self.__theta = np.linspace(0., angle, max(self.__image.shape), endpoint=True)
		self.__theta = np.linspace(0., angle, (angle+1)/step, endpoint=True)
		plt.figure(figsize=(10, 10))
		


	def run(self):
		start = time.clock()
		
		self.__acquisition()
		self.__showSinogram()
		self.__reconstruction()
		self.__showReconstruction()
		
		end = time.clock()
		print "time:", end-start
		
		plt.show()


	def __acquisition(self):
		self.__sinogram = self.__radon(self.__image, theta=self.__theta)
		#print self.__theta
		#print self.__sinogram

	def __reconstruction(self):
		self.__reconstruction_fbp = self.__iradon(self.__sinogram, theta=self.__theta)
		#self.__error = self.__reconstruction_fbp - self.__image
		#print('FBP rms reconstruction error: %.3g' % np.sqrt(np.mean(self.__error**2)))


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
		#plt.imshow(self.__reconstruction_fbp - self.__image, cmap=plt.cm.Greys_r, **self.__imkwargs)
		
		if show:
			plt.show()


	def __radon(self, image, theta=None):

		if image.ndim != 2:
			raise ValueError('The input image must be 2-D')
		if theta is None:
			raise ValueError('there is no theta in __radon!!!')

		height, width = image.shape
		diagonal = np.sqrt(height**2 + width**2)
		
		heightpad = np.ceil(diagonal - height)
		widthpad = np.ceil(diagonal - width)
		
		#padding image, so whole img is visible to "rays" while rotating
		padded_image = np.zeros((int(height + heightpad), int(width + widthpad)))
		y0, y1 = int(np.ceil(heightpad / 2)), int((np.ceil(heightpad / 2) + height))
		x0, x1 = int((np.ceil(widthpad / 2))), int((np.ceil(widthpad / 2) + width))
		padded_image[y0:y1, x0:x1] = image
		
		#plt.imshow(padded_image, cmap=plt.cm.Greys_r)
		#plt.show()
		
		sinogram = np.zeros((self.__detNum, len(theta)))

		h, w = padded_image.shape
		dh, dw = h // 2, w // 2
		
		shift0 = np.array([[1, 0, -dw],
						   [0, 1, -dh],
						   [0, 0, 1]])

		shift1 = np.array([[1, 0, dw],
						   [0, 1, dh],
						   [0, 0, 1]])


		def build_rotation(theta):
			T = -np.deg2rad(theta+90)

			R = np.array([[np.cos(T), -np.sin(T), 0],
						  [np.sin(T), np.cos(T), 0],
						  [0, 0, 1]])

			return shift1.dot(R).dot(shift0)
		

		for i in xrange(len(theta)):
			# apply transformation matrix to img, requires inverse of transformation matrix
			rotated = _warp_fast(padded_image, np.linalg.inv(build_rotation(-theta[i])))
			#rotated = warp(padded_image, np.linalg.inv(build_rotation(-theta[i])))		

			sinogram[:, i] = self.__radon_view(rotated)

		return sinogram
		
		
	def __radon_view(self, rotated):		
		height, width = rotated.shape
		
		emmitter_pos = (-self.__emmiterDistance, int(height/2) )
		detector_pos = [width + self.__detectorsDistance, int(height/2 - (self.__detNum*self.__detSize)/2 + np.floor(self.__detSize/2)) ]

		#Debug
		#print emmitter_pos, detector_pos

		view = np.zeros(self.__detNum)
		for i in xrange(0, self.__detNum):
			view[i] = self.__brasenham(emmitter_pos, detector_pos, rotated)
			detector_pos[1] += self.__detSize

		return view

	def __brasenham(self, p1, p2, image):
		s = 0
		
		x1, y1 = p1[0], p1[1]
		x2, y2 = p2[0], p2[1]
		x, y = x1, y1

		# direction of walking through pixels
		if (x1 < x2):
			xi = 1
			dx = x2 - x1
		else:
			xi = -1
			dx = x1 - x2

		# direction of walking through pixels
		if (y1 < y2):
			yi = 1
			dy = y2 - y1
		else:
			yi = -1
			dy = y1 - y2

		if x >= 0 and x < image.shape[1] and y >= 0 and y < image.shape[0]:
			s += image[y,x]

		if (dx > dy):
			ai = (dy - dx) * 2
			bi = dy * 2
			d = bi - dx

			while (x != x2 and x < image.shape[1]):
				if (d >= 0):
					x += xi
					y += yi
					d += ai
				else:
					d += bi
					x += xi
				if x >= 0 and x < image.shape[1] and y >= 0 and y < image.shape[0]:
					s += image[y,x]
		else:
			ai = ( dx - dy ) * 2
			bi = dx * 2
			d = bi - dy

			while (y != y2 and y < image.shape[0]):
				if (d >= 0):
					x += xi
					y += yi
					d += ai
				else:
					d += bi
					y += yi
				if x >= 0 and x < image.shape[1] and y >= 0 and y < image.shape[0]:
					s += image[y,x]

		return s

	def __iradon(self, radon_image, theta=None, output_size=None, fft_filter=False,
			   filter="ramp", interpolation="linear"):

		if radon_image.ndim != 2:
			raise ValueError('The input image must be 2-D')

		if theta is None:
			m, n = radon_image.shape
			theta = np.linspace(0, 180, n, endpoint=False)
		else:
			theta = np.asarray(theta)

		if len(theta) != radon_image.shape[1]:
			raise ValueError("The given ``theta`` does not match the number of projections in ``radon_image``.")

		th = (np.pi / 180.0) * theta
		#print th
		
		# if output size not specified, estimate from input radon image
		if not output_size:
			output_size = int(np.floor(np.sqrt((radon_image.shape[0])**2 / 2.0)))
		n = radon_image.shape[0]

		img = radon_image.copy()
		
		# resize image to next power of two for fourier analysis
		# speeds up fourier and lessens artifacts
		order = max(64., 2**np.ceil(np.log(2 * n) / np.log(2)))
		# zero pad input image
		img.resize((order, img.shape[1]))
		


		if fft_filter: #fft filter
			#construct the fourier filter
			f = fftshift(abs(np.mgrid[-1:1:2 / order])).reshape(-1, 1)
			w = 2 * np.pi * f
			# start from first element to avoid divide by zero
			if filter == "ramp":
				pass
			elif filter == "shepp-logan":
				f[1:] = f[1:] * np.sin(w[1:] / 2) / (w[1:] / 2)
			elif filter == "cosine":
				f[1:] = f[1:] * np.cos(w[1:] / 2)
			elif filter == "hamming":
				f[1:] = f[1:] * (0.54 + 0.46 * np.cos(w[1:]))
			elif filter == "hann":
				f[1:] = f[1:] * (1 + np.cos(w[1:])) / 2
			elif filter == None:
				f[1:] = 1
			else:
				raise ValueError("Unknown filter: %s" % filter)

			filter_ft = np.tile(f, (1, len(theta)))

			# apply filter in fourier domain
			projection = fft(img, axis=0) * filter_ft
			radon_filtered = np.real(ifft(projection, axis=0))
		
		else: #our filter
			filter_size = 10
			filter_tab = np.zeros((2*filter_size+1))
			for k in xrange(-filter_size,filter_size):
				if(k%2!=0):
					filter_tab[k+filter_size] = -4.0 / (np.pi**2 * k**2)  
			filter_tab[filter_size]=1
			
			radon_filtered = np.zeros((img.shape[0], img.shape[1]))
			for i in xrange(img.shape[1]):
				radon_filtered[:,i] = np.convolve(img[:,i],filter_tab, mode='same') 		

	


		# resize filtered image back to original size
		radon_filtered = radon_filtered[:radon_image.shape[0], :]
		
		#plt.show()
		#plt.imshow(radon_filtered, cmap=plt.cm.Greys_r)
		#plt.show()
		
		reconstructed = np.zeros((output_size, output_size))
		mid_index = np.ceil(n / 2.0)

		x = output_size
		y = output_size
		[X, Y] = np.mgrid[0.0:x, 0.0:y]
		xpr = X - int(output_size) // 2
		ypr = Y - int(output_size) // 2

		# reconstruct image by interpolation
		if interpolation == "nearest":
			for i in xrange(len(theta)):
				k = np.round(mid_index + xpr * np.sin(th[i]) - ypr * np.cos(th[i]))
				reconstructed += radon_filtered[ ((((k > 0) & (k < n)) * k) - 1).astype(np.int), i ]

		elif interpolation == "linear":
			for i in xrange(len(theta)):
				t = xpr * np.sin(th[i]) - ypr * np.cos(th[i])
				a = np.floor(t)
				b = mid_index + a
				b0 = ((((b + 1 > 0) & (b + 1 < n)) * (b + 1)) - 1).astype(np.int)
				b1 = ((((b > 0) & (b < n)) * b) - 1).astype(np.int)
				reconstructed += (t - a) * radon_filtered[b0, i] + (a - t + 1) * radon_filtered[b1, i]

		else:
			raise ValueError("Unknown interpolation: %s" % interpolation)


		return reconstructed * np.pi / (2 * len(th))
