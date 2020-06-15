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

class HeadPoseEstimationModel:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, threshold, device='CPU',extensions=None):
        '''
        DONE: Use this to set your instance variables.
        '''
        super(HeadPoseEstimationModel).__init__(model_name, threshold, device, extensions)


    def predict(self, image):
        '''
        DONE: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        image_final = self.preprocess_input(image)
        outputs = self.exec_network.infer({self.input_name:image_final})
        outputs_final = self.preprocess_output(outputs, image)

        return outputs_final


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


    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
    '''
        outs = [] 
        outs.append(outputs['angle_y_fc'].tolist()[0][0])
        outs.append(outputs['angle_p_fc'].tolist()[0][0])
        outs.append(outputs['angle_r_fc'].tolist()[0][0])

        return outs

