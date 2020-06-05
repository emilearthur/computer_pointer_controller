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

class FacialLandmarksModel:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        DONE: Use this to set your instance variables.
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

        self.model_structure = self.model_name 
        self.model_weights = self.model_name.split('.')[0]+'.bin'


    def load_model(self):
        '''
        DONE: You will need to complete this method.
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
        DONE: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        image_final = self.preprocess_input(image) 
        result = self.exec_network.infer({self.input_name:image_final})
        output = self.preprocess_output(result, image)
        
        left_eye_xmin = output['left_eye_x'] - 10 
        left_eye_xmax = output['left_eye_x'] + 10 
        left_eye_ymin = output['left_eye_y'] - 10 
        left_eye_ymax = output['left_eye_y'] + 10 

        right_eye_xmin = output['right_eye_x'] - 10 
        right_eye_xmax = output['right_eye_x'] + 10 
        right_eye_ymin = output['right_eye_y'] - 10 
        right_eye_ymax = output['right_eye_y'] + 10 

        left_eye = image[left_eye_xmin:left_eye_xmax, left_eye_ymin:left_eye_ymax]
        right_eye = image[right_eye_xmin:right_eye_xmax, right_eye_ymin:right_eye_ymin]

        eye_coords = [[left_eye_xmin, left_eye_ymin,left_eye_xmax, left_eye_ymax],
                        [right_eye_xmin, right_eye_ymin, right_eye_xmax, right_eye_ymax]]
        
        return left_eye, right_eye, eye_coords


    def check_model(self):
        pass


    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        image_cvt = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_final = cv2.resize(image_cvt, (self.input_shape[3], self.input_shape[2]))
        image_final = image_final.transpose((2,0,1)) 
        image_final = image_final.reshape(1,*image_final.shape)

        return image_final


    def preprocess_output(self, outputs, image):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
    '''
        outputs = outputs[self.output_names][0] 
        left_eye_x = int[outputs[0] * image.shape[1]]
        left_eye_y = int[outputs[1] * image.shape[0]]
        right_eye_x = int[outputs[2] * image.shape[1]]
        right_eye_y = int[outputs[3] * image.shape[0]]

        return {'left_eye_x_coords': left_eye_x, 'left_eye_y_coords': left_eye_y, 
                'right_eye_x_coords': right_eye_x, 'right_eye_y_coords': right_eye_y}