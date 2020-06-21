'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2 
from model import Model  

class FacialLandmarksModel(Model):
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, threshold=0.6, device='CPU',extensions=None):
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
        
        le_xmin = output['left_eye_x'] - 10 
        le_xmax = output['left_eye_x'] + 10 
        le_ymin = output['left_eye_y'] - 10 
        le_ymax = output['left_eye_y'] + 10 

        re_xmin = output['right_eye_x'] - 10 
        re_xmax = output['right_eye_x'] + 10 
        re_ymin = output['right_eye_y'] - 10 
        re_ymax = output['right_eye_y'] + 10 

        left_eye = image[le_ymin:le_ymax, le_xmin:le_xmax]
        right_eye = image[re_ymin:re_ymax, re_xmin:re_xmax]

        eye_coords = [[le_xmin, le_ymin,le_xmax, le_ymax],
                        [re_xmin, re_ymin, re_xmax, re_ymax]]
        
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
        h=image.shape[0]
        w=image.shape[1]

        outputs = outputs[self.output_names][0] 
        left_eye_x = int(outputs[0] * w)
        left_eye_y = int(outputs[1] * h)
        right_eye_x = int(outputs[2] * w)
        right_eye_y = int(outputs[3] * h)

        return {'left_eye_x': left_eye_x, 'left_eye_y': left_eye_y, 
                'right_eye_x': right_eye_x, 'right_eye_y': right_eye_y}