import os
import argparse
import cv2
import numpy as np
import sys
import importlib.util
from image import ImageFace


import math
from sklearn import neighbors
import os.path
import pickle
from PIL import Image, ImageDraw
from threading import Thread
import matplotlib.pyplot as plt
import time 
import re

FACES = 2

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--video', help='Name of the video file',
                    default='test.mp4')
					



args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph

VIDEO_NAME = args.video
min_conf_threshold = float(args.threshold)

pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
else:
    from tensorflow.lite.python.interpreter import Interpreter
	

# Get path to current working directory
CWD_PATH = os.getcwd()
VIDEO_PATH = os.path.join(CWD_PATH,VIDEO_NAME)
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)


interpreter = Interpreter(model_path=PATH_TO_CKPT)
interpreter.allocate_tensors()

iface = ImageFace(interpreter, min_conf_threshold)

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Open video file
video = cv2.VideoCapture(VIDEO_PATH)
imW = video.get(cv2.CAP_PROP_FRAME_WIDTH)
imH = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
frames_per_second = video.get(cv2.CAP_PROP_FPS)

def image_files_in_folder(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png)', f, flags=re.I)]



def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    """
    Trains a k-nearest neighbors classifier for face recognition.
    :param train_dir: directory that contains a sub-directory for each known person, with its name.
     (View in source code to see train_dir example tree structure)
     Structure:
        <train_dir>/
        ├── <person1>/
        │   ├── <somename1>.jpeg
        │   ├── <somename2>.jpeg
        │   ├── ...
        ├── <person2>/
        │   ├── <somename1>.jpeg
        │   └── <somename2>.jpeg
        └── ...
    :param model_save_path: (optional) path to save model on disk
    :param n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically if not specified
    :param knn_algo: (optional) underlying data structure to support knn.default is ball_tree
    :param verbose: verbosity of training
    :return: returns knn classifier that was trained on the given data.
    """
    X = []
    y = []

    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = iface.load_image(img_path)
            face_bounding_boxes = iface.face_locations(image)
            class_sample_image = image
            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                known_face = iface.face_encoding(image, face_bounding_boxes)[0]
                X.append(known_face)
                y.append(class_dir)

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance', )
    knn_clf.fit(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf

def load_model(model_path=None) :
    if model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    with open(model_path, 'rb') as f:
        knn_clf = pickle.load(f)
    return knn_clf 
    
    
def reco_faces(image, imgW, imgH) :
    frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    
    input_data = np.expand_dims(frame_resized, axis=0)
    
    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std
    
    
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
    
    locations = []
    if( len(scores) == 0 ) : 
        return []
    for i in range(len(scores)):
        if ((scores[i] > FACE_THRESHOLD) and (scores[i] <= 1.0)):
            
            top = int(max(1,(boxes[i][0] * imgH)))
            left  = int(max(1,(boxes[i][1] * imgW)))
            bottom= int(min(imgH,(boxes[i][2] * imgH)))
            right = int(min(imgW,(boxes[i][3] * imgW)))

            locations.append( (top, right, bottom, left) )
    return locations


def predict_faces(knn_clf, image, X_face_locations) :
    #print(f'Find Faces : {X_face_locations}')
    distance_threshold = 0.5
    if len(X_face_locations) == 0:
        return []
        
    #timestamp = time.time()
    # Find encodings for faces in the test iamge
    faces_encodings = iface.face_encodings(image, X_face_locations)
    #print(f'faces_encodings : {time.time()-timestamp}s')
    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=FACES)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

def show_prediction(image, predictions, is_showing=False) :

    #frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    for name, (top, right, bottom, left) in predictions:
    
       
        cv2.rectangle(image, (left,top), (right,bottom), (10, 255, 0), 4)
            
        # Draw label
        #object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
        label = name
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
        label_top = max(top, labelSize[1] + 10) # Make sure not to draw label too close to top of window
        cv2.rectangle(image, (left, label_top-labelSize[1]-10), (left+labelSize[0], label_top+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
        cv2.putText(image, label, (left, label_top-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
        
    #face_landmarks_list = face_recognition.face_landmarks(frame_rgb, model="large")

    #for face_landmarks in face_landmarks_list:

        # Print the location of each facial feature in this image
    #    for facial_feature in face_landmarks.keys():
    #        print("The {} in this face has the following points: {}".format(facial_feature, face_landmarks[facial_feature]))

        # Let's trace out each facial feature in the image with a line!
    #    for facial_feature in face_landmarks.keys():
            #print(f'point : {face_landmarks[facial_feature][0]}, {face_landmarks[facial_feature][1]}')
    #        if len(face_landmarks[facial_feature]) == 1 :
    #            cv2.line(image, face_landmarks[facial_feature][0], face_landmarks[facial_feature][0], (255, 255, 255), 2)
    #        else :
    #            cv2.line(image, face_landmarks[facial_feature][0], face_landmarks[facial_feature][1], (255, 255, 255), 2)
    if is_showing :
        cv2.imshow('Object detector', image)
        
    return image
    
print("Training KNN classifier...")
classifier = train("knn_examples/train", model_save_path="trained_knn_model.clf", n_neighbors=FACES)
print("Training complete!")

fnn_clf = load_model("trained_knn_model.clf")

   
OUTPUT_FILE_NAME = "output.avi" # mp4
OUTPUT_FILE_PATH = os.path.join(CWD_PATH, OUTPUT_FILE_NAME)
output_file = cv2.VideoWriter(
        filename=OUTPUT_FILE_NAME,
        fourcc=cv2.VideoWriter_fourcc(*"XVID"), # x264
        fps=float(frames_per_second),
        frameSize=(int(imW), int(imH)),
        isColor=True)

while(video.isOpened()):
	
    # Acquire frame and resize to expected shape [1xHxWx3]
    ret, frame = video.read()
    if not ret:
      print('Reached the end of the video!')
      break
      
      
    
    locations = reco_faces(frame, imW, imH)
    predictions = predict_faces(fnn_clf, frame, locations)
    prediction_frame = show_prediction(frame, predictions)
    
    output_file.write(prediction_frame)
    if cv2.waitKey(1) == ord('q'):
        break
    


# Clean up
video.release()
output_file.release()
cv2.destroyAllWindows()
	