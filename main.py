import CTsim
from skimage import data_dir



def main():
	file_path =  data_dir + "/phantom.png"
	#file_path =  "testR.png"
	a = CTsim.CTsimRadon(image_path=file_path, angle=180, step=10, detNum=500, detSize=5, emmDist=500, detDist=1500, fft=False, filter="ramp", mode=0)
	a.run()

if __name__ == '__main__':
	main()
