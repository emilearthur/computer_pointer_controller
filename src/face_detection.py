'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2 
from model import Model  

class FaceDetectionModel(Model):
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, threshold=0.6, extensions=None, device='CPU'):

        '''
        DONE: Use this to set your instance variables.
        '''
        super(FaceDetectionModel, self).__init__(model_name, threshold, extensions, device)


    def predict(self, image):
        '''
        DONE: You will need to complete this method.
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
