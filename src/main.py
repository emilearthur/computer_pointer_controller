import cv2 
import os 
import numpy as np 
import logging as log 
import time
import math  
import pandas as pd

from  face_detection import FaceDetectionModel 
from facial_landmarks_detection import FacialLandmarksModel 
from gaze_estimation import GazeEstimationModel 
from head_pose_estimation import HeadPoseEstimationModel 
from mouse_controller import MouseController 

from argparse import ArgumentParser 
from input_feeder import InputFeeder 

def build_argparser():
    """
    Parse commandline arguments 

    return: commandline  arguments 
    """
    parser = ArgumentParser() 

    parser.add_argument("-fd", "--face_dectection_model", required=True, type=str, help="Path to a face detection model xml file with a trained model.")
    parser.add_argument("-fl", "--face_landmarks_model", required=True, type=str, help="Path to a facial landmarks detection model xml file with a trained model.")
    parser.add_argument("-hp", "--head_pose_model", required=True, type=str, help="Path to a head pose model xml file with a trained model.")
    parser.add_argument("-ge", "--gaze_estimation_model", required=True, type=str,
                        help="Path to a gaze estimation model xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str, help="Path to image or video file or CAM")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,default=None, 
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d","--device", type=str, default="CPU", help="Specify the target device to infer on:")
    parser.add_argument("-pt", "--prob_threshold", type=float,default=0.6, help="Probability threshold for accurate detection (0.6 by default)")
    parser.add_argument("-flags", "--previewFlags", required=False, nargs='+', default=[],
                        help="Specify the flags from fd, fld, hp, ge like --flafs fd fl ge (seperate each flag by space)"
                        "for see the visualization of different model outputs of each frame,"
                        "fd for Face Detection Model, fl for Facial Landmark Detection model"
                        "hp for Head Pose Estimation Model and ge for Gaze Estimation Model")
    
    return parser



def main():
    # get command line args 
    args = build_argparser().parse_args()

    previewflags = args.previewFlags 
    logger = log.getLogger() 

    inputFilePath = args.input
    inputFeeder = None
    
    if inputFilePath.lower() == "cam":
        inputFeeder = InputFeeder("cam") 
    else:
        if not os.path.isfile(inputFilePath):
            logger.error("Unable to find specified video file") 
            exit(1)
        inputFeeder = InputFeeder("video", inputFilePath)
    
    modelPath = {'FaceDetectionModel':args.face_dectection_model,
                'FacialLandmarksModel': args.face_landmarks_model, 
                'HeadPoseEstimationModel':args.head_pose_model, 
                'GazeEstimationModel': args.gaze_estimation_model}
    
    for key in modelPath.keys():
        if not os.path.isfile(modelPath[key]):
            logger.error("Unable to find specific 'key' xml file")
            exit(1) 

    fdm = FaceDetectionModel(model_name = modelPath['FaceDetectionModel'], device = args.device,
                                            extensions = args.cpu_extension,threshold = args.prob_threshold)
    fldm = FacialLandmarksModel(model_name = modelPath['FacialLandmarksModel'], device = args.device, 
                                extensions = args.cpu_extension)
    gem = GazeEstimationModel(model_name = modelPath['GazeEstimationModel'], device = args.device, 
                            extensions = args.cpu_extension)
    hpem = HeadPoseEstimationModel(model_name = modelPath['HeadPoseEstimationModel'], device=args.device, 
                                extensions = args.cpu_extension)

    mc = MouseController("medium","fast") 


    # load input feeder 
    inputFeeder.load_data()

    # laod models and time taken to load models 
    data_capture = {}

    start_time = time.time()
    fdm.load_model() 
    fdm_load_time = time.time()
    fldm.load_model() 
    fldm_load_time = time.time() 
    hpem.load_model() 
    hpem_load_time = time.time()
    gem.load_model()
    gem_load_time = time.time()

    data_capture['FaceDetectionModel_time'] = round((fdm_load_time - start_time),1)
    data_capture['FacialLandmarksModel'] = round((fldm_load_time -fdm_load_time),1)
    data_capture['HeadPoseEstimationModel'] = round((hpem_load_time - fldm_load_time),1)
    data_capture['GazeEstimationModel'] = round((gem_load_time - hpem_load_time),1)

    counter = 0 
    start_infer_time = time.time() # time to start inference 
    for ret, frame in inputFeeder.next_batch():
        if not ret:
            break 
        counter+=1 
        if counter%5==0:
            cv2.imshow('video', cv2.resize(frame,(500,500))) 

        pressedKey = cv2.waitKey(60)
        face_coords, cropped_face = fdm.predict(frame.copy())
        
        if type(cropped_face) == int:
            logger.error("Unable to detect the face.")
            if key==27:
                break 
            continue 
        
        hpem_out = hpem.predict(cropped_face.copy())
        
        left_eye, right_eye, eye_coord = fldm.predict(cropped_face.copy())

        mouse_coord, gaze_vector = gem.predict(left_eye, right_eye, hpem_out) 

        if len(previewflags) != 0:
            preview_frame = frame.copy() 
            if 'fd' in previewflags:
                preview_frame = cropped_face
            if 'fld' in previewflags: 
                cv2.rectangle(cropped_face, (eye_coord[0][0]-10, eye_coord[0][1]-10), (eye_coord[0][2]+10, eye_coord[0][3]+10),
                            (0,255,0),3)
                cv2.rectangle(cropped_face, (eye_coord[1][0]-10, eye_coord[1][1]-10), (eye_coord[1][2]+10, eye_coord[1][3]+10),
                            (0,255,0),3)
            if 'hp' in previewflags:
                cv2.putText(preview_frame, "Pose Angles: yaw:{:.2f} | pitch:{:.2f} | roll:{:.2f}".format(hpem_out[0],
                                                                                                        hpem_out[1],
                                                                                                        hpem_out[2]),
                                                                                                        (10,20),
                                                                                                        cv2.FONT_HERSHEY_COMPLEX,
                                                                                                        0.25, (0,255,0), 1)
            if 'ge' in previewflags:
                x,y,w = int(gaze_vector[0]* 12), int(gaze_vector[1]*12), 160
                le = cv2.line(left_eye.copy(), (x-w, y-w), (x+w, y+w), (255,0,255),2)
                cv2.line(le, (x-w, y+w), (x+w, y-w), (255,0,255), 2) 
                re = cv2.line(right_eye.copy(), (x-w, y-w), (x+w, y+w), (255,0,255),2)
                cv2.line(re, (x-w, y+w), (x+w, y-w), (255,0,255), 2) 

                cropped_face[eye_coord[0][1]:eye_coord[0][3], eye_coord[0][0]:eye_coord[0][2]] = le
                cropped_face[eye_coord[1][1]:eye_coord[1][3], eye_coord[1][0]:eye_coord[1][2]] = re

            cv2.imshow("Visualization", cv2.resize(preview_frame,(500,500)))
        
        if counter%5 == 0:
            mc.move(mouse_coord[0], mouse_coord[1])
        if key == 27:
            break 

    infer_time = round((time.time() - start_infer_time),1)
    data_capture['Inference_time'] = infer_time

    df = pd.DataFrame.from_dict(data_capture,orient = 'index', columns=['time(secs)']) 
    df.to_csv("results.csv")

    logger.error("VideoStream Ended ...")
    cv2.destroyAllWindows() 
    inputFeeder.close()


if __name__ == '__main__':
    main()