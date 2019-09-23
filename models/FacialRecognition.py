import numpy as np
from Util import Util

class FacialRecognizer:

    def __init__(self, known_peoples_descriptors, known_peoples_names, model_path='./dlib_face_recognition_resnet_model_v1.dat', threshold=0.60, input_size=(512, 1024, 3), do_timing=False):
        self.threshold = threshold
        self.input_size = input_size
        self.image_size = None
        self.do_timing = do_timing
        self.result = None
        self.set_model(model_path)
        self.known_peoples_descriptors = known_peoples_descriptors
        self.known_peoples_names = known_peoples_names

    def set_model(self, model_path):
        from dlib import face_recognition_model_v1
        from pkg_resources import resource_filename
        self.model = face_recognition_model_v1(resource_filename(__name__, model_path)).compute_face_descriptor

    def get_result(self):
        return self.result

    def predict(self, image, faces, shapes):
        encodings = []
        for face, shape in zip(faces, shapes):
            rect = Util.list_to_rect(face)
            shape = Util.array_to_landmarks(shape, rect)
            encoding = self.model(image, shape)
            encodings.append(encoding)
        encodings = np.array(encodings)

        assigned_names = ["Unknown"] * len(faces)
        j = 0
        for encoding in encodings:
            enc_distances = [Util.face_encodings_distance(descriptor, encoding) for descriptor in self.known_peoples_descriptors]
            if enc_distances != []:
                min_distances = min(enc_distances)
                imin_distances = enc_distances.index(min_distances)
                if min_distances < 0.60:
                    assigned_names[j] = self.known_peoples_names[imin_distances]
            j += 1

        self.result = assigned_names

        return assigned_names
