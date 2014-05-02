import CTsim
from skimage import data_dir



def main():
	a = CTsim.CTsimRadon(image_path= data_dir + "/phantom.png", angle=180, step=1, detNum=250, detSize=2, emmDist=500, detDist=500)
	a.run()

if __name__ == '__main__':
	main()
