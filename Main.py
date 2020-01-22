##### IMPORTATIONS
import cv2
import pandas as pd
import numpy as np
import sys

from models.ObjectDetection import FaceDetector
from models.FacialLandmarksEstimation import FacialLandmarksEstimator
from models.ObjectTracking import MultiObjectTracker
from models.FacialRecognition import FacialRecognizer
from Pipeline import Pipeline


##### CONFIG
import yaml
cfg = yaml.load(open('./config.yml', 'r'))['Main']

known_peoples_pkl = str(cfg['known_peoples_pkl'])
detection_model = str(cfg['detection_model'])
det_tracking_model = str(cfg['det_tracking_model'])
tracking_model = str(cfg['tracking_model'])
rec_frequency = float(cfg['rec_frequency'])
video_file = int(cfg['video_file'])
resolution = tuple([int(x) for x in cfg['resolution']])
do_timing = bool(cfg['do_timing'])
#known_peoples_pkl = './data/known_peoples.pkl'
#detection_model = 'HOG'
#det_tracking_model = 'HOG'
#tracking_model = 'CSRT'
#rec_frequency = 0.01
#video_file = 0
#resolution = (640, 480)
#do_timing = True


##### RECOGNITION DATASET
# Import recognition dataset
df = pd.read_pickle(known_peoples_pkl)
known_peoples_descriptors = np.array([df['descriptor'].values[i] for i in range(df['descriptor'].values.shape[0])])
known_peoples_names = df['name'].values


##### MODELS
# Set models
face_detector = FaceDetector(model_path=detection_model, input_size=(resolution[1], resolution[0], 3), do_timing=do_timing)
facial_landmarks_estimator = FacialLandmarksEstimator(do_timing=do_timing)
multi_faces_tracker = MultiObjectTracker(model_path=tracking_model, do_timing=do_timing)
face_det_tracker = FaceDetector(model_path=det_tracking_model, input_size=(resolution[1], resolution[0], 3), do_timing=do_timing)
facial_recognizer = FacialRecognizer(known_peoples_descriptors, known_peoples_names, do_timing=do_timing)


##### VIDEO PARAMETERS
# Open video file
cap = cv2.VideoCapture(video_file)

# Check if video/camera opened successfully
if (cap.isOpened() == False):
    print('Error opening video stream or file')
    sys.exit()


###### MAIN LOOP
frame_number = 0
pipeline = Pipeline(regime='recognition', frequency=int(rec_frequency * 100))

# Read until video is completed
while(cap.isOpened()):

    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    frame_number += 1
    pipeline.frameNr += 1
    print('Frame number :', frame_number)

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, resolution)



    # Process frame
    if pipeline.regime == 'recognition':
        print('     ---- Recognition stage ----     ')

        print('Face detection')
        face_bboxes, face_bboxes_confidences = face_detector.predict(image)
        res_face_detection = face_detector.get_result()
        #face_detector.visualize(image, face_bboxes, face_bboxes_confidences, color=(255, 0, 0))

        print('Initialize tracking...')
        multi_faces_tracker.initialize(image, face_bboxes)

        print('Facial landmarks estimation')
        facial_landmarks, facial_landmarks_confidences = facial_landmarks_estimator.predict(image, face_bboxes, face_bboxes_confidences)
        res_facial_landmarks_estimation = facial_landmarks_estimator.get_result()
        #facial_landmarks_estimator.visualize(image, facial_landmarks, facial_landmarks_confidences, color=(255, 0, 0))

        print('Facial recognition')
        names = facial_recognizer.predict(image, face_bboxes, facial_landmarks)
        res_facial_recognition = facial_recognizer.get_result()

        pipeline.update(res_face_detection.bounding_boxes, res_facial_landmarks_estimation.facial_landmarks, res_facial_recognition)

        pipeline.visualize(image)
        cv2.putText(image, 'Recognition stage', (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))


    elif pipeline.regime == 'detection':
        print('     ---- Detection stage   ----     ')

        print('Face detection')
        face_bboxes, face_bboxes_confidences = face_det_tracker.predict(image)
        res_face_detection = face_det_tracker.get_result()
        #face_detector.visualize(image, face_bboxes, face_bboxes_confidences, color=(0, 0, 255))

        print('Initialize tracking...')
        multi_faces_tracker.initialize(image, face_bboxes)

        print('Facial landmarks estimation')
        facial_landmarks, facial_landmarks_confidences = facial_landmarks_estimator.predict(image, face_bboxes, face_bboxes_confidences)
        res_facial_landmarks_estimation = facial_landmarks_estimator.get_result()
        #facial_landmarks_estimator.visualize(image, facial_landmarks, facial_landmarks_confidences, color=(0, 0, 255))

        pipeline.match(res_face_detection.bounding_boxes, res_facial_landmarks_estimation.facial_landmarks)

        pipeline.visualize(image)
        cv2.putText(image, 'Detection stage', (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))


    elif pipeline.regime == 'tracking':
        print('     ---- Tracking stage    ----     ')

        print('Face tracking')
        face_bboxes, face_bboxes_confidences = multi_faces_tracker.update(image)
        #face_detector.visualize(image, face_bboxes, face_bboxes_confidences, color=(0, 255, 0))

        print('Facial landmarks estimation')
        facial_landmarks, facial_landmarks_confidences = facial_landmarks_estimator.predict(image, face_bboxes, face_bboxes_confidences)
        #facial_landmarks_estimator.visualize(image, facial_landmarks, facial_landmarks_confidences, color=(0, 255, 0))

        pipeline.match()

        pipeline.visualize(image)
        cv2.putText(image, 'Tracking stage', (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))

    pipeline.update_regime()



    # Show and/or write frame
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# When everything done, release the video capture object
cap.release()
# Closes all the frames
cv2.destroyAllWindows()
