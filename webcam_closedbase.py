# IMPORTS
import numpy as np
import h5py
import cv2
import functions
import utils

# IMPORT DATASET
known_peoples_descriptors_file = h5py.File('./ressources/datasets/known_peoples.hdf5', 'r')
known_peoples_descriptors = known_peoples_descriptors_file["known_peoples_descriptors"] # (n, 128)

known_peoples_names_file = open('./ressources/datasets/known_peoples_names.txt', 'r')
known_peoples_names = []
for name in known_peoples_names_file:
    known_peoples_names.append(name) # (n,)
known_peoples_names = [line.strip() for line in known_peoples_names]

# WEBCAM LIVE
video_capture = cv2.VideoCapture(0)
assigned_names = []

while True:

    # Get the frame
    _, frame = video_capture.read()
    rgb_frame = frame[:, :, ::-1]

    # Update the condition on the activation of the recognition stage
    chance_percentage = 0.10
    random_number = np.random.randint(0, 10)
    activate_recognition = random_number < (chance_percentage * 10)

    # Recognition stage
    if activate_recognition:

        print("Recognition stage")

        # Face locations
        faces = functions.face_locations(rgb_frame, model="HOG")

        # Face descriptors
        encodings = functions.face_descriptors(rgb_frame, faces)

        # Assign names
        assigned_names = ["Unknown"] * len(faces)
        j = 0
        for encoding in encodings:
            enc_distances = [utils.face_encodings_distance(descriptor, encoding) for descriptor in known_peoples_descriptors]
            if enc_distances != []:
                min_distances = min(enc_distances)
                imin_distances = enc_distances.index(min_distances)
                if min_distances < 0.60:
                    assigned_names[j] = known_peoples_names[imin_distances]
            j += 1


        # Draw on the frame
        utils.draw_on_frame(frame, faces, assigned_names)

    # Tracking stage
    else:

        print("Tracking stage")

        # Face locations
        new_faces = functions.face_locations(rgb_frame, model="MTCNN")

        if assigned_names == []:
            assigned_names = ["Unknown"] * len(new_faces)
            faces = new_faces

        else:
            #faces <-> assigned_names
            #new_faces <->assigned_new_names
            assigned_new_names = ["Unknown"] * len(new_faces)
            track_distances = utils.global_distances(new_faces, faces)
            # ième ligne jème colonne de track_distances = distance entre new_faces[i] et faces[j] (distance = IoU)
            # donc track_distances[i] est la ième ligne dont les composantes sont la distance entre new_faces[i] et tous les visages détectés précédemment
            for i in range(len(new_faces)):
                min_iou = min(track_distances[i])
                imin_iou = track_distances[i].index(min_iou)
                assigned_new_names[i] = assigned_names[imin_iou]

            # Need to compute Global Nearest Neighbor
            # For future projects : Gating, PDAF, JPDAF, MHT, MCMCDA

            assigned_names = assigned_new_names
            faces = new_faces

        # Draw on the frame
        utils.draw_on_frame(frame, faces, assigned_names)


    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
