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

def output_boxes(image, point_one, point_two, color=(0,255,0)): # head dectection flag
	cv2.rectangle(image, point_one, point_two, color, 1)

def output_text(image, text, org):
	color=(255,255,0)
	cv2.putText(image,text, org, 0, 0.8,color)


def bound_boxes(image, eye_coord, height, width, x0, y0):
    color=(255,255,0)

    ex1, ey1, ex2, ey2 = eye_coord

    x1, x2 = ex1-int(width/2), ex1+int(width/2)
    y1, y2 = ey1-int(height/2 + 1), ey1+int(height/2 + 1)



    point_one = (x0+x1, y0+y1)
    point_two = (x0+x2, y0+y2)

    output_boxes(image, point_one, point_two,color)





# code source: https://knowledge.udacity.com/questions/171017
def draw_axes(frame, center_of_face, yaw, pitch, roll, scale = 50, focal_length=950):
    yaw *= np.pi / 180.0
    pitch *= np.pi / 180.0
    roll *= np.pi / 180.0
    cx = int(center_of_face[0])
    cy = int(center_of_face[1])
    r_x = np.array([[1, 0, 0],
                    [0, math.cos(pitch), -math.sin(pitch)],
                    [0, math.sin(pitch), math.cos(pitch)]])
    r_y = np.array([[math.cos(yaw), 0, -math.sin(yaw)],
                    [0, 1, 0],
                    [math.sin(yaw), 0, math.cos(yaw)]])
    r_z = np.array([[math.cos(roll), -math.sin(roll), 0],
                    [math.sin(roll), math.cos(roll), 0],
                    [0, 0, 1]])

    r = r_z @ r_y @ r_x

    # source and reference https://www.learnopencv.com/rotation-matrix-to-euler-angles/
    camera_matrix = build_camera_matrix(center_of_face, focal_length)
    xaxis = np.array(([1 * scale, 0, 0]), dtype='float32').reshape(3, 1)
    yaxis = np.array(([0, -1 * scale, 0]), dtype='float32').reshape(3, 1)
    zaxis = np.array(([0, 0, -1 * scale]), dtype='float32').reshape(3, 1)
    zaxis1 = np.array(([0, 0, 1 * scale]), dtype='float32').reshape(3, 1)
    o = np.array(([0, 0, 0]), dtype='float32').reshape(3, 1)
    o[2] = camera_matrix[0][0]
    xaxis = np.dot(r, xaxis) + o
    yaxis = np.dot(r, yaxis) + o
    zaxis = np.dot(r, zaxis) + o
    zaxis1 = np.dot(r, zaxis1) + o
    xp2 = (xaxis[0] / xaxis[2] * camera_matrix[0][0]) + cx
    yp2 = (xaxis[1] / xaxis[2] * camera_matrix[1][1]) + cy
    p2 = (int(xp2), int(yp2))
    cv2.line(frame, (cx, cy), p2, (0, 0, 255), 2)
    xp2 = (yaxis[0] / yaxis[2] * camera_matrix[0][0]) + cx
    yp2 = (yaxis[1] / yaxis[2] * camera_matrix[1][1]) + cy
    p2 = (int(xp2), int(yp2))
    cv2.line(frame, (cx, cy), p2, (0, 255, 0), 2)
    xp1 = (zaxis1[0] / zaxis1[2] * camera_matrix[0][0]) + cx
    yp1 = (zaxis1[1] / zaxis1[2] * camera_matrix[1][1]) + cy
    p1 = (int(xp1), int(yp1))
    xp2 = (zaxis[0] / zaxis[2] * camera_matrix[0][0]) + cx
    yp2 = (zaxis[1] / zaxis[2] * camera_matrix[1][1]) + cy
    p2 = (int(xp2), int(yp2))
    cv2.line(frame, p1, p2, (255, 0, 0), 2)
    cv2.circle(frame, p2, 3, (255, 0, 0), 2)
    return frame


def build_camera_matrix(center_of_face, focal_length):
    cx = int(center_of_face[0])
    cy = int(center_of_face[1])
    camera_matrix = np.zeros((3, 3), dtype='float32')
    camera_matrix[0][0] = focal_length
    camera_matrix[0][2] = cx
    camera_matrix[1][1] = focal_length
    camera_matrix[1][2] = cy
    camera_matrix[2][2] = 1
    return camera_matrix





def build_argparser():
    """
    Parse commandline arguments 

    return: commandline  arguments 
    """
    parser = ArgumentParser() 

    parser.add_argument("-fd","--face_dectection_model", required=True, type=str, help="Path to a face detection model.")
    parser.add_argument("-fl","--face_landmarks_model", required=True, type=str, help="Path to a facial landmarks detection model.")
    parser.add_argument("-hp","--head_pose_model", required=True, type=str, help="Path to a head pose model.")
    parser.add_argument("-ge","--gaze_estimation_model", required=True, type=str,
                        help="Path to a gaze estimation model.")
    parser.add_argument("-i","--input", required=True, type=str, help="Path to image or video file or CAM")
    parser.add_argument("-e","--cpu_extension", required=False, type=str,default=None, 
                        help="absolute path of cpu extension.")
    parser.add_argument("-d","--device", default="CPU", type=str, help="Device to run inference on")
    parser.add_argument("-pt", "--prob_threshold", default=0.6, type=float,  help="Probability threshold for accurate detection (0.6 by default)")
    parser.add_argument("-outputflag", "--preview",default=False,action="store_true",help="Displays output of imtermediate models")


    return parser



def main():
    # get command line args 
    args = build_argparser().parse_args()

    logger = log.getLogger() 

    type_input = args.input

    if type_input == "CAM":
        inputFeeder = InputFeeder("cam")
    else: 
        inputFeeder = InputFeeder("video", args.input)

    inputFeeder.load_data()

    mc = MouseController("medium","fast") 

    fdm = FaceDetectionModel(model_name = args.face_dectection_model, device = args.device,
                                            extensions = args.cpu_extension,threshold = args.prob_threshold)
    fldm = FacialLandmarksModel(model_name = args.face_landmarks_model, device = args.device, 
                                extensions = args.cpu_extension)
    gem = GazeEstimationModel(model_name = args.gaze_estimation_model, device = args.device, 
                            extensions = args.cpu_extension)
    hpem = HeadPoseEstimationModel(model_name = args.head_pose_model, device=args.device, 
                                extensions = args.cpu_extension)
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

    data_capture['FaceDetectionModel_loadtime'] = round((fdm_load_time - start_time)*1000,3)
    data_capture['FacialLandmarksModel_loadtime'] = round((fldm_load_time -fdm_load_time)*1000,3)
    data_capture['HeadPoseEstimationModel_loadtime'] = round((hpem_load_time - fldm_load_time)*1000,3)
    data_capture['GazeEstimationModel_loadtime'] = round((gem_load_time - hpem_load_time)*1000,3)
    
    for flag, frame in inputFeeder.next_batch():
        if not flag:
            break 

        pressedKey = cv2.waitKey(60)

        start_infer_time = time.time() # time to start inference 
        face_coords, face_img = fdm.predict(frame) 
        fdm_infertime = time.time()

        if face_coords == 0: # if face not detected
            continue

        hpem_out = hpem.predict(face_img)
        hpem_infertime = time.time()

        left_eye, right_eye, eye_coord = fldm.predict(face_img)
        fldm_infertime = time.time()
        
        if left_eye.all() == 0 or  right_eye.all()==0: # if eye are not detected
            continue


        mouse_coord, gaze_vector = gem.predict(left_eye, right_eye, hpem_out) 
        gem_infertime = time.time()


        if args.preview:
            output_boxes(frame, (face_coords[0],face_coords[1]),(face_coords[2],face_coords[3]))


            bound_boxes(frame, eye_coord, 45, 25, face_coords[0], face_coords[1])


            text = "Yaw: {:.2f}, Pitch: {:+.2f}, Roll: {:.2f}".format(hpem_out[0],hpem_out[1], hpem_out[2])

            output_text(frame, text,(100,100))

            h=frame.shape[0]
            w=frame.shape[1]

            center_of_face = (h/2, w/2,0) 

            draw_axes(frame, center_of_face, hpem_out[0], hpem_out[1], hpem_out[2], scale = 50, focal_length=950)

            cv2.imshow('video', cv2.resize(frame,(500,500))) 


        cv2.imshow('video', cv2.resize(frame,(500,500))) 
        mc.move(mouse_coord[0], mouse_coord[1])

        if pressedKey == 27:
            break 



    data_capture['FaceDetectionModel_Inferencetime'] = round((fdm_infertime - start_infer_time)*1000,3)
    data_capture['HeadPoseEstimationModel_Inferencetime'] = round((hpem_infertime - fdm_infertime)*1000,3)
    data_capture['FacialLandmarksModel_Inferencetime'] = round((fldm_infertime - hpem_infertime)*1000,3)
    data_capture['GazeEstimationModel_Inferencetime'] = round((gem_infertime - fldm_infertime)*1000,3)

    total_time = round((time.time() - start_infer_time)*1000,3)
    data_capture['Total_time'] = total_time

    df = pd.DataFrame.from_dict(data_capture,orient = 'index', columns=['time(msecs)']) 
    df.to_csv("results.csv")

    logger.error("Video has ended...")
    cv2.destroyAllWindows() 
    inputFeeder.close()


if __name__ == '__main__':
    main()