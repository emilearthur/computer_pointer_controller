    
'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2 
import os 
import sys 
import logging as log 
import math 
from openvino.inference_engine import IECore 

class Model_X:
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
        self.plugin = None 
        self.exec_network = None  
        self.infer_request = None 
        self.input_name = None 
        self.input_shape = None 
        self.output_name = None 
        self.output_shape = None 

        self.model_structure = self.model_name 
        self.model_weights = self.model_name.split('.')[0]+'.bin'



        

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.plugin = IECore() 
        self.network = self.plugin.read_network(model=self.model_structure, weights = self.model_weights) 
        supported_layers  = self.plugin.query_network(network=self.network, device_name=self.device) 
        not_supported_layer = [layers for layers in self.network.layers.keys() if layers not in supported_layers]
        

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        raise NotImplementedError

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        raise NotImplementedError

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
    '''
        raise NotImplementedError
