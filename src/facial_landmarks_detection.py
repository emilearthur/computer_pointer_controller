'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2
from model import Model  
from openvino.inference_engine import IECore 

class FacialLandmarksModel(Model):
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, threshold, device='CPU',extensions=None):
        '''
        DONE: Use this to set your instance variables.
        '''
        super(FacialLandmarksModel,self).__init__(model_name, threshold, device, extensions)


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