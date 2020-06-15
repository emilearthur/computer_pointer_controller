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
from openvino.inference_engine import IECore, IENetwork 


class Model:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, threshold, device='CPU',extensions=None):
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
        self.output_name = None 
        self.output_shape = None 
        self.threshold = threshold

    def load_model(self):
        '''
        DONE: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        try:
            self.model_structure = self.model_name 
            self.model_weights = self.model_name.split('.')[0]+'.bin'
            self.core = IECore() 
            self.model = self.core.read_network(model=self.model_structure, weights = self.model_weights) 
        except Exception as e:
            raise ValueError("Could not Initialise the network. Enter Correct Model path ?")


        supported_layers  = self.core.query_network(network=self.network, device_name=self.device) 
        not_supported_layers = [layer for layer in self.network.layers.keys() if layer not in supported_layers]

        if len(not_supported_layers) > 0 and self.device:
            if not self.extensions == None:
                self.core.add_extension(self.extensions, self.device) 
                supported_layers  = self.core.query_network(network=self.network, device_name=self.device) 
                not_supported_layer = [layers for layers in self.network.layers.keys() if layers not in supported_layers]
                if len(not_supported_layer) > 0:
                    sys.exit(1)
            else: 
                sys.exit(1)
        

        self.exec_network = self.core.load_network(network=self.network, device_name=self.device, num_requests=1) 

        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape