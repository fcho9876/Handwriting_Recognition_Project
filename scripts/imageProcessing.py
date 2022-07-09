# File name: imageProcessing.py
# Author: Francis Cho
# Project Version: 2.0
# Description: Contains image processing functions
# Python Version: 3.1

import torch.nn
from PIL import Image, ImageOps, ImageFilter
import numpy as np

import torch
import cv2

class imageProcessing():
    def __init__(self):
        super().__init__()

        # process image before fed into a train model
    def process_input_image_dif(self):
        # original image is stored in a 400 x 400 pixel 
        #img = img.resize((200, 200))  # could improve speed of prediction by reducing image size
        img = Image.open('images/loadedimage.png')

        
        # apply gaussian blur/fliter with sigma = 1
        image_blurred = img.filter(ImageFilter.GaussianBlur(radius = 1))
        image_blurred.save('images/(b)_Gaussian_Blur.png')

        # extract ROI from our initial inputted image
        # use nested for loop to go by row and column to find blank space (contains zeros only)

        image_ROI = image_blurred.convert('L')              # convert image to black and white
        image_ROI = np.array(image_ROI)                     # convert to a numpy array data
        image_ROI = np.invert(image_ROI)                    # inverts the array

        # set up dimensions of image shape and define arrays to store position of zeros
        numRows, numCols = image_ROI.shape
        zero_row_array = []
        zero_col_array = []
        for i in range(0, numRows - 1):
            for j in range(0, numCols):
                if (np.count_nonzero(image_ROI[i, :])):
                    pass    # do nothing if current index is a non-zero
                else:
                    zero_row_array.append(i)
                
                if (np.count_nonzero(image_ROI[:, j])):
                    pass    # do nothing if current index is a non-zero
                else:
                    zero_col_array.append(j)

        # remove zeros to get our ROI
        image_ROI = np.delete(image_ROI, tuple(zero_row_array), axis = 0)
        image_ROI = np.delete(image_ROI, tuple(zero_col_array), axis = 1)
   
        # preserve aspect ratio by setting both dimensions to the higher dimension
        temp_img = Image.fromarray(image_ROI, 'L')
        temp_img.save('images/(c)_ROI_Extraction.png')

        pixel_width, pixel_height = temp_img.size
        new_aspect_ratio = max(pixel_width, pixel_height)
        image_ROI = temp_img.resize((new_aspect_ratio, new_aspect_ratio))

        # Resize image to 26 by 26 to add a 2 pixel border
        image_ROI = image_ROI.resize((26,26))

        # Create a blank 28,28 black image
        image_centered = Image.new('L', (28,28))

        # Paste the 20,20 in the center to make the completed 28,28
        image_centered.paste(image_ROI, (1,1))
        image_centered.save('images/(d)_Centered_Frame.png')

        # flip and rotate 90 degrees
        newImg_flip = ImageOps.mirror(image_centered)
        newImg_rotate = newImg_flip.rotate(90)

        # final processed output image
        newImg_rotate.save('./images/(e)_Resized.png')

        # adjust for correct dtype
        image_adjust = np.array(newImg_rotate).astype(np.float32) / 255
        image_adjust_Tensor = torch.from_numpy(image_adjust)
        
        # convert 2D tensor to a 4D input by adding two dimensions for batch loading
        image_adjust_Tensor = torch.unsqueeze(image_adjust_Tensor, 0)
        image_adjust_Tensor = torch.unsqueeze(image_adjust_Tensor, 0)

        return image_adjust_Tensor


    def process_capture_image_dif(self):
        img = Image.open('images/captured_image.png')

        # convert to a numpy array data
        img = np.array(img)                  

        # apply gaussian blur and convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        cv2.imwrite('images/first_blur.png', image_blurred)

        # apply adaptive thresholding
        image_thresh = cv2.adaptiveThreshold(image_blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 81, 10)
        #cv2.imshow("Gaussian Adaptive Thresholding", thresh)
        cv2.imwrite('images/(b)Captured_Gaussian_Blur.png', image_thresh)

        # extract ROI from our initial inputted image
        # use nested for loop to go by row and column to find blank space (contains zeros only)
        image_ROI = image_thresh

        # set up dimensions of image shape and define arrays to store position of zeros
        numRows, numCols = image_ROI.shape
        zero_row_array = []
        zero_col_array = []
        for i in range(0, numRows - 1):
            for j in range(0, numCols):
                if (np.count_nonzero(image_ROI[i, :])):
                    pass    # do nothing if current index is a non-zero
                else:
                    zero_row_array.append(i)
                
                if (np.count_nonzero(image_ROI[:, j])):
                    pass    # do nothing if current index is a non-zero
                else:
                    zero_col_array.append(j)

        # remove zeros to get our ROI
        image_ROI = np.delete(image_ROI, tuple(zero_row_array), axis = 0)
        image_ROI = np.delete(image_ROI, tuple(zero_col_array), axis = 1)
   
        # preserve aspect ratio by setting both dimensions to the higher dimension
        temp_img = Image.fromarray(image_ROI, 'L')
        temp_img.save('images/(c)Captured__ROI_Extraction.png')

        pixel_width, pixel_height = temp_img.size
        new_aspect_ratio = max(pixel_width, pixel_height)
        image_ROI = temp_img.resize((new_aspect_ratio, new_aspect_ratio))

        # Resize image to 26 by 26 to add a 2 pixel border
        image_ROI = image_ROI.resize((26,26))

        # Create a blank 28,28 black image
        image_centered = Image.new('L', (28,28))

        # Paste the 20,20 in the center to make the completed 28,28
        image_centered.paste(image_ROI, (1,1))
        image_centered.save('images/(d)Captured__Centered_Frame.png')

        # flip and rotate 90 degrees
        newImg_flip = ImageOps.mirror(image_centered)
        newImg_rotate = newImg_flip.rotate(90)

        # final processed output image
        newImg_rotate.save('./images/(e)Captured__Resized.png')

        # adjust for correct dtype
        image_adjust = np.array(newImg_rotate).astype(np.float32) / 255
        image_adjust_Tensor = torch.from_numpy(image_adjust)
        
        # convert 2D tensor to a 4D input by adding two dimensions for batch loading
        image_adjust_Tensor = torch.unsqueeze(image_adjust_Tensor, 0)
        image_adjust_Tensor = torch.unsqueeze(image_adjust_Tensor, 0) 

        return image_adjust_Tensor       


