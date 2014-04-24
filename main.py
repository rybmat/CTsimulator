import CTsim
from skimage import data_dir



def main():
	#a = CTsim.CTsimRadon(image_path="test.png", angle=180)
	a = CTsim.CTsimRadon(image_path=data_dir + "/phantom.png", angle=180)
	a.run()

if __name__ == '__main__':
	main()
