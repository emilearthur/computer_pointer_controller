    
'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2 
from model import Model  
import math


class GazeEstimationModel(Model):
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, threshold=0.6, extensions=None, device='CPU'):
        '''
        DONE: Use this to set your instance variables.
        '''
        super(GazeEstimationModel,self).__init__(model_name, threshold, extensions, device)




    def predict(self, left_eye, right_eye, hpa):
        '''
        DONE: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
    
        left_eye_final, right_eye_final = self.preprocess_input(left_eye, right_eye) 
        outputs = self.exec_network.infer({'left_eye_image':left_eye_final,
                                            'right_eye_image':right_eye_final,
                                            'head_pose_angles':hpa})
        
        mouse_coordinate, gaze_vector = self.preprocess_output(outputs, hpa)

        return mouse_coordinate, gaze_vector


    def check_model(self):
        pass


    def preprocess_input(self, left_eye, right_eye):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        left_eye_final = cv2.resize(left_eye, (60,60))
        right_eye_final = cv2.resize(right_eye, (60,60))

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
        outputs_vec = outputs[self.output_names][0]
        roll_value = hpa[2]

        cos_theta = math.cos(roll_value* math.pi / 180.0)
        sin_theta = math.sin(roll_value*math.pi / 180.0)

        x_value = outputs_vec[0] * cos_theta + outputs_vec[1] * sin_theta
        y_value = outputs_vec[1] * cos_theta - outputs_vec[0] * sin_theta

        return (x_value, y_value), outputs_vec