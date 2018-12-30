# IMPORTS
from pkg_resources import resource_filename
import numpy as np
import dlib
from mtcnn.mtcnn import MTCNN
import utils


# MODELS
# Convert saved models (.dat) into functions which can be called later

face_locations_hog = dlib.get_frontal_face_detector()
# face_locations_hog :
#       Input :
#       Output :

face_locations_mtcnn = MTCNN().detect_faces

face_landmarks = dlib.shape_predictor(resource_filename(__name__, "ressources/models/shape_predictor_68_face_landmarks.dat"))
# face_landmarks :
#       Input :
#       Output :

face_encodings = dlib.face_recognition_model_v1(resource_filename(__name__, "ressources/models/dlib_face_recognition_resnet_model_v1.dat")).compute_face_descriptor
# face_descriptors :
#       Input :
#       Output :


# FUNCTIONS

def face_locations(image, model="HOG"):
    # Given an image, return a list of coordinates of detected faces
    # Input : an image (numpy array), the name of the detector
    # Output : list of face boxes coordinates (list of list of 4 elements)

    faces = []

    if model == "HOG":

        print("Model used for face detection : HOG")

        f = face_locations_hog(image, 1)
        print(len(f), "face(s) detected.")

        for _, face in enumerate(f):
            faces.append(utils.rect_to_list(face))

    elif model == "MTCNN":

        print("Model used for face detection : MTCNN")

        f = face_locations_mtcnn(image)
        for i in range(len(f)):
            f[i] = f[i]['box']
            f[i][2] = f[i][0] + f[i][2]
            f[i][3] = f[i][1] + f[i][3]
            faces.append(f[0])

    else:

        print("Error in the choice of the model, models available : HOG, MTCNN\n")

    return faces

def face_descriptors(image, faces):
    # input : a frame (numpy matrix) and face locations (list of coordinates)
    # output : list of descriptors associated to locations (list of column vectors)

    descriptors = []

    for face in faces:
        face = utils.list_to_rect(face)
        shape = face_landmarks(image, face)
        descriptor = face_encodings(image, shape)
        descriptors.append(np.array(descriptor))

    return descriptors
