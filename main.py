import CTsim
from skimage import data_dir



def main():
	a = CTsim.CTsimRadon(image_path= data_dir + "/phantom.png", angle=180, step=1, detNum=150, detSize=3, emmDist=750, detDist=750)
	a.run()

if __name__ == '__main__':
	main()
