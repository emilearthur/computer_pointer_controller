    
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

class GazeEstimation:
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




    def predict(self, left_eye, right_eye, hpa):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        left_eye_final, right_eye_final = self.preprocess_input(left_eye, right_eye) 
        outputs = self.exec_network.infer({'left_eye_image':left_eye_final,'right_eye_image':right_eye_final,
                                            'head_pose_angles':hpa})
        
        mouse_coordinate, gaze_vector = self.preprocess_output(outputs, hpa)


    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, left_eye, right_eye):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        #left_eye_resized = cv2.resize(left_eye, (self.input_shape[3], self.input_shape[2]))
        #right_eye_resized = cv2.resize(left_eye, (self.input_shape[3], self.input_shape[2]))
        #left_eye_final = np.transpose(np.expand_dims(left_eye_resized, axis=0), (0,3,1,2))
        #right_eye_final = np.transpose(np.expand_dims(right_eye_resized, axis=0), (0,3,1,2))
        left_eye_final = cv2.resize(left_eye, (self.input_shape[3], self.input_shape[2]))
        right_eye_final = cv2.resize(right_eye, (self.input_shape[3], self.input_shape[2]))

        left_eye_final = left_eye_final.transpose((2,0,1))
        right_eye_final = right_eye_final.transpose((2,0,1))

        left_eye_final = left_eye_final.reshape(1, *left_eye_final.shape)
        right_eye_final = right_eye_final.reshape(1, *right_eye_final.shape)

        return left_eye_final, right_eye_final

    def preprocess_output(self, outputs, hpa):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
    '''
        outputs_vec = outputs[self.output_names[0]].tolist()[0]
        roll_value = hpa[2]

        cos_theta = math.cos(roll_value* math.pi / 180.0)
        sin_theta = math.sin(roll_value*math.pi / 180.0)

        x_value = outputs_vec[0] * cos_theta + outputs_vec[1] * sin_theta
        y_value = outputs_vec[1] * cos_theta - outputs_vec[0] * sin_theta

        return (x_value, y_value), outputs_vec

        
