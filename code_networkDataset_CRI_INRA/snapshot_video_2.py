# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 10:39:04 2019

@author: Aurel
"""

# -*- coding: utf-8 -*-

     
 
import cv2
import os

image_folder = 'ima2'
video_name = 'V2_addition.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 1, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()    
