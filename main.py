import CTsim
from skimage import data_dir



def main():
	a = CTsim.CTsimRadon(image_path= data_dir + "/phantom.png", angle=180, step=1, detNum=227, detSize=3)
	a.run()

if __name__ == '__main__':
	main()
