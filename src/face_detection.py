'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import cv2 
import os 
import sys 
import logging as log 
import math 
import numpy as np 
from openvino.inference_engine import IECore 

class FaceDetectionModel:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, threshold, device='CPU',extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_name = model_name 
        self.device = device 
        self.extensions = extensions
        self.network = None 
        self.core = None 
        self.exec_network = None  
        self.infer_request = None 
        self.input_name = None 
        self.input_shape = None 
        self.output_names = None 
        self.output_shape = None 
        self.threshold = threshold

        self.model_structure = self.model_name 
        self.model_weights = self.model_name.split('.')[0]+'.bin'

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.core = IECore() 
        self.network = self.core.read_network(model=self.model_structure, weights = self.model_weights) 
        supported_layers  = self.core.query_network(network=self.network, device_name=self.device) 
        not_supported_layer = [layers for layers in self.network.layers.keys() if layers not in supported_layers]

        if len(not_supported_layer) > 0 and self.device:
            if not self.extensions == None:
                self.core.add_extension(self.extensions, self.device) 
                supported_layers  = self.core.query_network(network=self.network, device_name=self.device) 
                not_supported_layer = [layers for layers in self.network.layers.keys() if layers not in supported_layers]
                if len(not_supported_layer) > 0:
                    sys.exit(1)
            else: 
                sys.exit(1)
        

        self.exec_network = self.core.load_network(network=self.network, device_name=self.device, num_requests=1) 
        self.input_name = [i for i in self.network.inputs.keys()]
        self.input_shape = self.network.inputs[self.input_name[1]].shape
        self.output_names = [i for i in self.network.outputs.keys()]

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        image_final = self.preprocess_input(image)
        outputs = self.exec_network.infer({self.input_name:image_final})
        face_coord = self.preprocess_output(outputs, image)

        if len(face_coord) == 0:
            return 0,0
        first_face_coord = face_coord[0]
        cropped_face = image[first_face_coord[1]:first_face_coord[3],
                        first_face_coord[0]:first_face_coord[2]]

        return first_face_coord, cropped_face        


    def check_model(self):
        pass

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        image_final = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        image_final = image_final.transpose((2,0,1)) 
        image_final = image_final.reshape(1,*image_final.shape) 

        return image_final

    def preprocess_output(self, outputs, image):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
    '''
        face_coords = [] 
        results = outputs[self.output_names][0][0] 
        for face in results:
            conf = face[2]
            if conf >= self.threshold:
                xmin = int(face[3] * image.shape[1]) 
                ymin = int(face[4] * image.shape[0]) 
                xmax = int(face[5] * image.shape[1])
                ymax = int(face[6] * image.shape[0])
                
                face_coords.append([xmin, ymin,xmax, ymax])

        return face_coords