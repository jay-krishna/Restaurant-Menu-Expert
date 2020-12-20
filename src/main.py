#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import sys
from skimage.filters import threshold_otsu
from skimage.morphology import disk
from skimage.morphology import dilation
from PIL import Image
import pytesseract
import os
import argparse


from resturant_menu import resturant_menu_expert


if __name__ == "__main__":

	ap = argparse.ArgumentParser()
	ap.add_argument("-n", "--img_path", required=True, help="Path Of the Image Of menu")
	ap.add_argument("-d", "--max_dist", required=True, help="Maximum edit Distance Allowed For OCR Correction Algorithm")
	
	args = vars(ap.parse_args())
	
	path = args['img_path']
	max_dis = int(args['max_dist'])
	
	print(max_dis*2)
	
	resturant_menu_expert(path ,max_dis)
	
