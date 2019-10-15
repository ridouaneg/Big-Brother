import time
from PIL import Image
import numpy as np
import dlib


class Util:

    @staticmethod
    def load_image(image_path):
        """Load an image given its path thanks to the Pillow Image module then
        convert it to a numpy array.

        Parameters
        ----------
        image_path : str
            The path to the image

        Returns
        ------
        image : numpy array
            The image as a numpy array
        """
        image = Image.open(image_path)
        image = np.array(image)
        return image

    @staticmethod
    def rect_to_list(rect):
        xmin, ymin, xmax, ymax = rect.left(), rect.top(), rect.right(), rect.bottom()
        return [xmin, ymin, xmax, ymax]

    @staticmethod
    def list_to_rect(l):
        xmin, ymin, xmax, ymax = l[0], l[1], l[2], l[3]
        return dlib.rectangle(xmin, ymin, xmax, ymax)

    @staticmethod
    def landmarks_to_array(lm):
        arr = np.array([[lm.part(j).x, lm.part(j).y] for j in range(68)])
        return arr

    @staticmethod
    def array_to_landmarks(arr, rect):
        points = [dlib.point(x, y) for (x, y) in arr]
        lm = dlib.full_object_detection(rect, points)
        return lm

    @staticmethod
    def face_encodings_distance(face_encoding1, face_encoding2):
        # Given two face descriptors, return the distance between them
        # input : two descriptors to compare (two 128-dim vectors)
        # output : the distance between those descriptors (the L1-norm)
        return np.linalg.norm(face_encoding1 - face_encoding2)

    @staticmethod
    def bbox_distance_iou(bbox1, bbox2):
        xmin1, ymin1, xmax1, ymax1 = bbox1
        xmin2, ymin2, xmax2, ymax2 = bbox2

        xA = max(xmin1, xmin2)
        yA = max(ymin1, ymin2)
        xB = min(xmax1, xmax2)
        yB = min(ymax1, ymax2)

        intersection = max(0, xB - xA) * max(0, yB - yA)

        bbox1_area = (xmax1 - xmin1) * (ymax1 - ymin1)
        bbox2_area = (xmax2 - xmin2) * (ymax2 - ymin2)

        union = bbox1_area + bbox2_area - intersection

        IoU = intersection / (union + 1e-5)

        return IoU
