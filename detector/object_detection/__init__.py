#author: Cataldo Cianciaruso
#
#This component perform an Object Detection on input image.
#It's based upon ImageAI library (https://github.com/OlafenwaMoses/ImageAI)

import os
import sys
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import numpy as np
import urllib.request

stdout = sys.stdout
sys.stdout = open(os.devnull, 'w')
import tensorflow as tf

import keras

from keras import backend as K
from keras.layers import Input
sys.stdout = stdout

from object_detection.keras_retinanet.models.resnet import resnet50_retinanet
from object_detection.keras_retinanet.utils.image import read_image_binary, preprocess_image, resize_image

execution_path = os.getcwd()

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = False
    return tf.Session(config=config)


class Object_Detector:

    def __init__(self):
        self.__modelType = ""
        self.modelPath = os.path.join(os.path.join(execution_path, "models") , "resnet50_coco_best_v2.0.1.h5")
        self.__modelPathAdded = True
        self.__modelLoaded = False
        self.__model_collection = []
        self.__custom_objects=self._only_objects()

        # Instance variables for RetinaNet Model
        self.__input_image_min = 1333
        self.__input_image_max = 800

        self.numbers_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
                                 6: 'train',
                                 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign',
                                 12: 'parking meter',
                                 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
                                 20: 'elephant',
                                 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
                                 27: 'tie',
                                 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball',
                                 33: 'kite',
                                 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard',
                                 38: 'tennis racket',
                                 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
                                 45: 'bowl',
                                 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot',
                                 52: 'hot dog',
                                 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant',
                                 59: 'bed',
                                 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote',
                                 66: 'keyboard',
                                 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
                                 72: 'refrigerator',
                                 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear',
                                 78: 'hair drier',
                                 79: 'toothbrush'}


        if not(os.path.isfile(self.modelPath)):
            print("Downloading model..")
            urllib.request.urlretrieve("https://dl.dropboxusercontent.com/s/qxza99du1mkxwmf/resnet50_coco_best_v2.0.1.h5", mpath)
            print("..done!")

            
        self.__modelType = "retinanet"
        self.loadModel(detection_speed="fast")
        self.sess = K.get_session()


    def loadModel(self, detection_speed="normal"):
       

        if (self.__modelType == "retinanet"):
            if (detection_speed == "normal"):
                self.__input_image_min = 800
                self.__input_image_max = 1333
            elif (detection_speed == "fast"):
                self.__input_image_min = 400
                self.__input_image_max = 700
            elif (detection_speed == "faster"):
                self.__input_image_min = 300
                self.__input_image_max = 500
            elif (detection_speed == "fastest"):
                self.__input_image_min = 200
                self.__input_image_max = 350
            elif (detection_speed == "flash"):
                self.__input_image_min = 100
                self.__input_image_max = 250

        if (self.__modelLoaded == False):
            if (self.__modelType == ""):
                raise ValueError("You must set a valid model type before loading the model.")
            elif (self.__modelType == "retinanet"):
                model = resnet50_retinanet(num_classes=80)
                model.load_weights(self.modelPath)
                self.__model_collection.append(model)
                self.__modelLoaded = True
            



    def _only_objects(self):

         custom_objects = {}
        
         labels = ["person", "bicycle", "car", "motorcycle", "airplane",
                         "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
                         "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
                         "zebra",
                         "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
                         "snowboard",
                         "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                         "tennis racket",
                         "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
                         "orange",
                         "broccoli", "carrot", "hot dog", "pizza", "donot", "cake", "chair", "couch", "potted plant",
                         "bed",
                         "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                         "microwave",
                         "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
                         "hair dryer",
                         "toothbrush"]

         for label in labels:
             custom_objects[label] = "valid"
         
         custom_objects["person"]="invalid"   
         return custom_objects
     


    def detect_objects(self, input_image, minimum_percentage_probability=50):

        if (self.__modelLoaded == False):
            raise ValueError("You must call the loadModel() function before making object detection.")
        elif (self.__modelLoaded == True):
            try:
                output_objects_array = []
                detected_objects_image_array = []
                image = read_image_binary(input_image)
                image = preprocess_image(image)
                image, scale = resize_image(image, min_side=self.__input_image_min, max_side=self.__input_image_max)
                model = self.__model_collection[0]
                _, _, detections = model.predict_on_batch(np.expand_dims(image, axis=0))
                predicted_numbers = np.argmax(detections[0, :, 4:], axis=1)
                scores = detections[0, np.arange(detections.shape[1]), 4 + predicted_numbers]
                detections[0, :, :4] /= scale

                min_probability = minimum_percentage_probability / 100

                for index, (label, score), in enumerate(zip(predicted_numbers, scores)):
                    if score < min_probability:
                        continue

                    if (self.__custom_objects != None):
                        check_name = self.numbers_to_names[label]

                    if (self.__custom_objects[check_name] == "invalid"):
                        continue

                    detection_details = detections[0, index, :4].astype(int)
                    x, y, px, py=detection_details
                    each_object_details = {}
                    each_object_details["name"] = self.numbers_to_names[label]
                    each_object_details["accuracy"] = score * 100
                    each_object_details["coordinates"] = detection_details
                    output_objects_array.append(each_object_details)

                return output_objects_array
            except:
                raise ValueError(
                    "Ensure you specified correct input image, input type, output type and/or output image path ")
