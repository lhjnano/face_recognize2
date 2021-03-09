import os
import argparse
import cv2
import numpy as np
import sys
import glob
import importlib.util
import dlib 

try:
    import face_recognition_models
except Exception:
    print("Please install `face_recognition_models` with this command before using `face_recognition`:\n")
    print("pip install git+https://github.com/ageitgey/face_recognition_models")
    quit()


class ImageFace :
    

    def __init__(self, model, threshold) :
        self.interpreter = model
        self.input_details = model.get_input_details()
        self.output_details = model.get_output_details()
        self.floating_model = (self.input_details[0]['dtype'] == np.float32)
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]
        self.min_conf_threshold = threshold
                
        face_detector = dlib.get_frontal_face_detector()
        predictor_68_point_model = face_recognition_models.pose_predictor_model_location()
        self.pose_predictor_68_point = dlib.shape_predictor(predictor_68_point_model)
        
        face_recognition_model = face_recognition_models.face_recognition_model_location()
        self.face_encoder = dlib.face_recognition_model_v1(face_recognition_model)

    def load_image(self, path) :
        image = cv2.imread(path)
        
        return image
        
    def face_locations(self, image):
    
        input_mean = 127.5
        input_std = 127.5
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imH, imW, _ = image.shape 
        image_resized = cv2.resize(image_rgb, (self.width, self.height))
        input_data = np.expand_dims(image_resized, axis=0)
        
            # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if self.floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std
    
    
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        locations = []
        
        # Retrieve detection results
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0] # Bounding box coordinates of detected objects
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0] # Confidence of detected objects
        
        for i in range(len(scores)):
            if ((scores[i] > self.min_conf_threshold) and (scores[i] <= 1.0)):
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                
                locations.append([xmin, ymin, xmax, ymax])
                
        return locations 
        
        
    
    def _css_to_rect(css):
        """
        Convert a tuple in (top, right, bottom, left) order to a dlib `rect` object
        :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
        :return: a dlib `rect` object
        """
        return dlib.rectangle(css[0], css[1], css[2], css[3])


    def _raw_face_landmarks(face_image, face_locations):
        num_jitters = 1
        face_locations = [_css_to_rect(face_location) for face_location in face_locations]

        pose_predictor = self.pose_predictor_68_point

        return [pose_predictor(face_image, face_location) for face_location in face_locations]
        
    def face_encoding(self, image, locations):
        #여러 얼굴 이미지 반환 
        raw_landmarks = _raw_face_landmarks(image, locations)
        return [np.array(self.face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)) for raw_landmark_set in raw_landmarks]








