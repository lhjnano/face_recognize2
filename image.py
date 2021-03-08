import os
import argparse
import cv2
import numpy as np
import sys
import glob
import importlib.util


class ImageFace :
    
    def __init__(self, model) :
        self.input_details = model.get_input_details()
        self.output_details = model.get_output_details()
        
    def load_image(self, path) :
        image = cv2.imread(path)
        
        return image
        
    def face_locations(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imH, imW, _ = image.shape 
        image_resized = cv2.resize(image_rgb, (width, height))
        input_data = np.expand_dims(image_resized, axis=0)
        
        interpreter.set_tensor(self.input_details[0]['index'], input_data)
        interpreter.invoke()
        
        locations = []
        
        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
        scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
        
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                
                locations.append(xmin, ymin, xmax, ymax)
                
        return locations 
        
        
    def face_encoding(self, image, locations):
        #여러 얼굴 이미지 반환 
        location = location[0]
        xmin = location[0]
        ymin = location[1]
        xmax = location[2]
        ymax = location[3]
        return image[xmin:ymin, xmax:ymax]
                