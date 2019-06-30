# encoding=utf8
#Object detection using Tensorflow and Keras.
#
#author: Cataldo Cianciaruso

import sys
sys.stdout.flush        
import os
import warnings
import tensorflow as tf

warnings.filterwarnings("ignore")
execution_path = os.getcwd()
print("Loading Object Detector......")
from detector.object_detection import Object_Detector

class Detector_Server:


    def __init__(self):
        self._faces=[]
        self._objects=[]
        self._emotions=[]
        self._object_detector = Object_Detector()

    def get_result(self):
        result="OBJ"
        for obj in self._objects:
            result+=obj+","
        
        return result[:-1]
   


    def detect(self, file):
        
        self._objects=[]
        print("\nAnalyzing objects...")
        detections = self._object_detector.detect_objects(file, minimum_percentage_probability=30)
        
        if(len(detections)>0):
            det=""
            for eachObject in detections:
                
                det=str(str(eachObject["name"])+"-"+str(eachObject["coordinates"][0])+" "+
                        str(eachObject["coordinates"][1])+" "+str(eachObject["coordinates"][2])+" "+
                        str(eachObject["coordinates"][3]))

                self._objects.append(det)
        
        print("done!")        
              


if __name__== "__main__":
    pass



