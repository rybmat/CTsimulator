import CTsim
from skimage import data_dir



def main():
	a = CTsim.CTsimRadon(image_path= data_dir + "/phantom.png", angle=360, step=1, detNum=100, detSize=4, emmDist=450, detDist=450)
	a.run()

if __name__ == '__main__':
	main()
