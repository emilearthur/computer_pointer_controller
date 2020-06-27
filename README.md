# Computer Pointer Controller

## `Introduction`
Computer Pointer Controller App is an application used to control the movement of mouse using the gaze estimation model.

The gaze estimation model is used to estimate the gaze of the users's eyes and it's feed into the `pyautogui` module to change the mouse pointer position.

## Project Set Up and Installation
Projcet source structure is a show below.

```
pointer-controller
├── README.md
├── bin
│   └── demo.mp4
    └── pipeline.png
    └── tree.jpg
├── intel
│   ├── face-detection-adas-binary-0001
│   │   └── FP32-INT1
│   │       ├── face-detection-adas-binary-0001.bin
│   │       └── face-detection-adas-binary-0001.xml
│   ├── gaze-estimation-adas-0002
│   │   ├── FP16
│   │   │   ├── gaze-estimation-adas-0002.bin
│   │   │   └── gaze-estimation-adas-0002.xml
│   │   ├── FP16-INT8
│   │   │   ├── gaze-estimation-adas-0002.bin
│   │   │   └── gaze-estimation-adas-0002.xml
│   │   └── FP32
│   │       ├── gaze-estimation-adas-0002.bin
│   │       └── gaze-estimation-adas-0002.xml
│   ├── head-pose-estimation-adas-0001
│   │   ├── FP16
│   │   │   ├── head-pose-estimation-adas-0001.bin
│   │   │   └── head-pose-estimation-adas-0001.xml
│   │   ├── FP16-INT8
│   │   │   ├── head-pose-estimation-adas-0001.bin
│   │   │   └── head-pose-estimation-adas-0001.xml
│   │   └── FP32
│   │       ├── head-pose-estimation-adas-0001.bin
│   │       └── head-pose-estimation-adas-0001.xml
│   └── landmarks-regression-retail-0009
│       ├── FP16
│       │   ├── landmarks-regression-retail-0009.bin
│       │   └── landmarks-regression-retail-0009.xml
│       ├── FP16-INT8
│       │   ├── landmarks-regression-retail-0009.bin
│       │   └── landmarks-regression-retail-0009.xml
│       └── FP32
│           ├── landmarks-regression-retail-0009.bin
│           └── landmarks-regression-retail-0009.xml
├── requirements.txt

└── src
    ├── face_detection.py
    ├── facial_landmarks_detection.py
    ├── gaze_estimation.py
    ├── head_pose_estimation.py
    ├── input_feeder.py
    ├── main.py
    ├── model.py
    └── mouse_controller.py
    └── result.csv
```

### Setup
Prerequisites
* You need to install openvino successfully. See [link](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux.html) for installation.
* THis project depends also on other additional libraries mentioned in `requirments.txt` file. Install the dependencies as shown below.

`pip install -r requirements.txt`

#### Step 1 
Clone the repository https://github.com/emilearthur/computer_pointer_controller

#### Step 2 
Initialize openVINO environment

`source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5`

#### Step 3 
Download the following models with their precisions by using openVINO model downloader
##### Face Detection Model 

`python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name face-detection-adas-binary-0001 `
##### Facial Landmarks Detection Model 

`python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name landmarks-regression-retail-0009 --precisions FP16,FP16-INT8,FP32`
##### Head Pose Estimation Model 

`python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name head-pose-estimation-adas-0001 --precisions FP16,FP16-INT8,FP32`
##### Gaze Estimation Model 

`python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name gaze-estimation-adas-0002 --precisions FP16,FP16-INT8,FP32`
## Demo
How to run:

* To run app [main.py] (On CPU)

`cd src/`

```
python main.py -fd <Path of xml file of face detection model> \
-fl <Path of xml file of facial landmarks detection model> \
-hp <Path of xml file of head pose estimation model> \
-ge <Path of xml file of gaze estimation model> \
-i <Path of input video file or enter cam for taking input video from webcam>
```

* To run app [main.py] on GPU

```
python main.py -fd <Path of xml file of face detection model> \
-fl <Path of xml file of facial landmarks detection model> \
-hp <Path of xml file of head pose estimation model> \
-ge <Path of xml file of gaze estimation model> \
-i <Path of input video file or enter cam for taking input video from webcam>
-d GPU
```

* To run app [main.py] on FPGA

```
python main.py -fd <Path of xml file of face detection model> \
-fl <Path of xml file of facial landmarks detection model> \
-hp <Path of xml file of head pose estimation model> \
-ge <Path of xml file of gaze estimation model> \
-i <Path of input video file or enter cam for taking input video from webcam>
-d HETERO:FPGA,CPU
```

Command Line Arguments for running the app

* -fl(required): Specify the path of Face detection model's xml file.
* -hp (required): Specify the path of Head Pose Estimation model's xml file.
* -ge (required): Specify the path of Gaze Estimation model's xml file.
* -i (required): Specify the path of the input video or enter cam for taking inpput video from webcam.
* -d (optional): Specify the target device to infer the video file on the model. Devices supported include CPU, GPU, FPGA and MYRIAD.
* -l (optional): Specify the absolute path of cpu extension.
* -pt (optional): Specify the probability threshold for face detection model.
* -flags (optional): Specify the flags from fd, fldm, hp and ge if you want to visualize the output of corresponding model of each frame. Note: Write flags with space seperation Example: --flags fd fld hp

## Documentation
`Pipeline`:  This project makes use of four pre-trained models provided by Intel's OpenVINO toolkits. The image below, shows the flow between then.

![Pipeline](bin/pipeline.png)

* [Face Detection](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)
 * [Head Position Estimation](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
 * [Facial Landmark Detection](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)
 * [Gaze Estimation Model](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)

## Benchmarks
| Model                            | Precision | Load Time    | Inference Time | Pre-process Time |
| -------------------------------- | --------- | -----------: | -------------- | ---------------- |  
| face-detection-adas-binary-0001  | FP32-INT1 | 212.75 msecs | 9.99 msecs     | 0.55 msecs       |
| landmarks-regression-retail-0009 | FP16      | 108.77 msecs | 0.55 msecs     | 0.05 msecs       |
|                                  | FP16-INT8 | 132.88 msecs | 0.42 msecs     |                  |
|                                  | FP32      | 105.91 msecs | 0.54 msecs     |                  |
| head-pose-estimation-adas-0001   | FP16      | 129.37 msecs | 1.22 msecs     | 0.06 msecs       |
|                                  | FP16-INT8 | 207.70 msecs | 1.18 msecs     |                  |
|                                  | FP32      |  97.98 msecs | 1.35 msecs     |                  |
| gaze-estimation-adas-0002        | FP16      | 139.08 msecs | 1.58 msecs     | 0.05 msecs       |
|                                  | FP16-INT8 | 237.18 msecs | 1.39 msecs     |                  |
|                                  | FP32      | 122.49 msecs | 1.85 msecs     |                  |

## Results
As seen from the benchmark table above, there is difference in inference time across difference precision like FP32, FP16 and INT8 models. The reason for maybe due to quantization technics used for optimization which makes use of INT8 weights instead of FLOATS, which in-turn increase inference time but reduces accuracy. Thus accuracy and inference being in the order of `FP32>FP16>INT8`. However, It was discovered that the model size is inversely proportional to precision i.e. lighter model has lower precision.

## Edge Cases
For multiple face detected, only one face is extracted and used to control the mouse pointer.