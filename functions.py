# IMPORTS
from pkg_resources import resource_filename
import dlib
import utils
import numpy as np
from mtcnn.mtcnn import MTCNN


# MODELS

# face detectors
face_locations_hog = dlib.get_frontal_face_detector()
face_detector_mtcnn = MTCNN()

# facial landmarks and descriptor extractor
face_landmarks_68_points = dlib.shape_predictor(resource_filename(__name__, "ressources/models/shape_predictor_68_face_landmarks.dat"))
face_descriptor_resnet = dlib.face_recognition_model_v1(resource_filename(__name__, "ressources/models/dlib_face_recognition_resnet_model_v1.dat"))


# FUNCTIONS

# face locations
def face_locations(image, model="HOG"):
    # input : an image (numpy array)
    # output : list of face boxes coordinates (list of list of 4 elements)

    faces = []

    if model == "MTCNN":

        print("Model used for face detection : MTCNN")

        faces = face_detector_mtcnn.detect_faces(image)
        for i in range(len(faces)):
            faces[i] = faces[i]['box']
            faces[i][2] = faces[i][0] + faces[i][2]
            faces[i][3] = faces[i][1] + faces[i][3]

    elif model == "HOG":

        print("Model used for face detection : HOG")

        f = face_locations_hog(image)
        for _, face in enumerate(f):
            faces.append(utils.rect_to_list(face))

    else:

        print("Error in the choice of the model, models available : HOG, MTCNN")

    return faces

# extract descriptors
def face_descriptors(frame, face_locations):
    # input : a frame (numpy matrix) and face locations (list of coordinates)
    # output : list of descriptors associated to locations (list of column vectors)

    face_descriptors = []

    for face_location in face_locations:

        shape = face_landmarks_68_points(frame, utils.list_to_rect(face_location))
        descriptor = face_descriptor_resnet.compute_face_descriptor(frame, shape)
        face_descriptors.append(np.array(descriptor))

    return face_descriptors

"""
# tracking
def tracking(face_locations, frame, prev_name, prev_coord):
    #print('Noms précédents : {}'.format(prev_name))
    #print('Coordonnées précédentes : {}'.format(prev_coord))
    coord = [(face_locations[i][0], face_locations[i][1]) for i in range(len(face_locations))]
    name = [""]*len(coord)
    for k in range(len(coord)):
        distances = [dist(coord[k], prev_coord[j]) for j in range(len(prev_coord))]
        if distances != []:
            min_distances = min(distances)
            imin_distances = distances.index(min_distances)
            name[k] = prev_name[imin_distances]
            prev_name.remove(prev_name[imin_distances])
            prev_coord.remove(prev_coord[imin_distances])
        else:
            name = ["Unknown"]*len(coord)
    #print('Nouveaux noms ; {}'.format(name))
    #print('Nouvelles coordonnées ; {}'.format(coord))
    for j in range(len(face_locations)):
        left, top, right, bottom = face_locations[j][0], face_locations[j][1], face_locations[j][2], face_locations[j][3]
        print(left, top, right, bottom)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        print(name[j])
        cv2.putText(frame, name[j], (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    return name, coord
"""
