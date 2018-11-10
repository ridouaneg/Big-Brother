# IMPORTS
import h5py
import cv2
import numpy as np
import functions
import utils

# IMPORT DATASET
known_peoples_descriptors_file = h5py.File('./ressources/known_peoples/known_peoples_descriptors.hdf5', 'r')
known_peoples_descriptors = known_peoples_descriptors_file["known_peoples_descriptors"] # (n, 128)

known_peoples_names_file = open('./ressources/known_peoples/known_peoples_names.txt', 'r')
known_peoples_names = []
for name in known_peoples_names_file:
    known_peoples_names.append(name) # (n,)
known_peoples_names = [line.strip() for line in known_peoples_names]

# WEBCAM LIVE
video_capture = cv2.VideoCapture(0)

#individus = -1
#name = []

#tmp_track = 1
#nb_track = 10

while True:

    # Get the frame
    _, frame = video_capture.read()
    rgb_frame = frame[:, :, ::-1]

    # Update the condition on the activation of the recognition stage
    chance_percentage = 1.
    random_number = np.random.randint(0, 10)
    activate_recognition = random_number < (chance_percentage * 10)

    # Recognition stage
    if activate_recognition:

        print("Recognition stage")

        # Face locations

        face_locations = functions.face_locations(rgb_frame, "MTCNN")
        nb_faces = len(face_locations)
        print("Nombre de personnes détectées : ", nb_faces)

        # Face descriptors
        face_descriptors = functions.face_descriptors(rgb_frame, face_locations)

        # Assign names
        assigned_names = ["Unknown"] * len(face_locations)
        j = 0
        for face_descriptor in face_descriptors:
            distances = [utils.distance(descriptor, face_descriptor) for descriptor in known_peoples_descriptors]
            if distances != []:
                min_distances = min(distances)
                imin_distances = distances.index(min_distances)
                if min_distances < 0.55:
                    assigned_names[j] = known_peoples_names[imin_distances]
            j += 1

        # Draw on the frame
        utils.draw_on_frame(frame, face_locations, assigned_names)


    # Tracking stage
    else:

        print("Tracking stage")





    """
    face_locations = functions.face_locations(rgb_frame, "HOG")
    tmp = len(face_locations)

    if tmp > individus or tmp_track == nb_track+1:

        # recognition
        print(' ')
        print('Phase de reconnaissance')
        print('Nombre de personnes détectées : {}'.format(len(face_locations)))
        name, coord = functions.recognition_closed(frame, face_locations, known_peoples_encodings, known_peoples_labels)
        tmp_track = 1
        #print('Nom des personnes détectées : {}'.format(name))
        #print('Coordonnées des personnes détectées : {}'.format(coord))
        print(' ')

    else:

        # tracking
        print('Phase de tracking : {} / {}'.format(tmp_track, nb_track))
        print('Nombre de personnes détectées par le HOG : {}'.format(len(face_locations)))
        face_locations = functions.face_locations(rgb_frame, "MTCNN")
        print('Nombre de personnes détectées par le MTCNN : {}'.format(len(face_locations)))
        #name, coord = functions.tracking(face_locations, frame, name, coord)
        name, coord = functions.tracking_v2(face_locations, frame, name, coord)
        tmp_track += 1
        #print('Nom des personnes détectées : {}'.format(name))
        #print('Coordonnées des personnes détectées : {}'.format(coord))

    individus = tmp
    """

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
