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
    def __init__(self, model_name, threshold, extensions=None, device='CPU'):
        '''
        DONE: Use this to set your instance variables.
        '''
        self.model_name = model_name 
        self.device = device 
        self.extensions = extensions
        self.model = None 
        self.core = None 
        self.exec_network = None  
        self.infer_request = None 
        self.input_name = None 
        self.input_shape = None 
        self.output_name = None 
        self.output_shape = None 
        self.threshold = threshold

        self.model_structure = self.model_name 
        self.model_weights = self.model_name.split('.')[0]+'.bin'
        
    def load_model(self):
        '''
        DONE: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
            
        self.core = IECore() 
        self.model = IENetwork(model=self.model_structure, weights = self.model_weights) 
        supported_layers  = self.core.query_network(network=self.model, device_name=self.device) 
        not_supported_layers = [layers for layers in self.model.layers.keys() if layers not in supported_layers]

        if len(not_supported_layers) > 0 and self.device == 'CPU':
            print("Unsupported layers found:{}".format(not_supported_layers))
            if not self.extensions==None:
                print("Adding CPU Extension")
                self.core.add_extension(self.extensions, self.device) 
                supported_layers  = self.core.query_network(network=self.model, device_name=self.device) 
                not_supported_layers = [layers for layers in self.model.layers.keys() if layers not in supported_layers]
                if len(not_supported_layers) > 0:
                    print("Unsupported layers found")
                    exit(1) 
            else:
                print("Give the path of cpu extension") 
                exit(1)


        self.exec_network = self.core.load_network(network=self.model, device_name=self.device, num_requests=1) 

        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_names = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_names].shape 



        
         

        

        