import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from Util import Util
from models.ObjectDetection import BoundingBox

class Person:

    def __init__(self, person_id, face_bounding_box=None, facial_landmarks=None):
        self.person_id = person_id
        self.face_bounding_box_path = [face_bounding_box]
        self.facial_landmarks_path = [facial_landmarks]

class Pipeline:

    def __init__(self, humans=[], regime='recognition', frequency=12):
        self.humans = humans
        self.regime = regime
        self.frameNr = 0
        self.frequency = frequency

    def update_regime(self):
        if self.frameNr % self.frequency == 0:
            self.regime = 'recognition'
        else:
            self.regime = 'detection'

    def update(self, new_face_bboxes, new_facial_landmarks, new_ids):
        humans = []

        previous_ids = [self.humans[i].person_id for i in range(len(self.humans))]

        for k in range(len(new_ids)):
            if new_ids[k] in previous_ids:
                ind = previous_ids.index(new_ids[k])
                self.humans[ind].face_bounding_box_path.append(new_face_bboxes[k])
                self.humans[ind].facial_landmarks_path.append(new_facial_landmarks[k])
                humans.append(self.humans[ind])
            else:
                new_human = Person(new_ids[k], new_face_bboxes[k], new_facial_landmarks[k])
                humans.append(new_human)

        self.humans = humans

    def match(self, new_face_bboxes, new_facial_landmarks):

        distances = np.array([[Util.bbox_distance_iou(new_face_bboxes[i].bounding_box, self.humans[j].face_bounding_box_path[-1].bounding_box) \
                                                    for j in range(len(self.humans))]
                                                    for i in range(len(new_face_bboxes))])

        if distances.shape[0] == 0:
            row_ind, col_ind = [], []
        else:
            row_ind, col_ind = linear_sum_assignment(- distances)

        unmatched_tracking_indices = []
        for j in range(len(self.humans)):
            # unmatched tracking
            if j not in col_ind:
                unmatched_tracking_indices.append(j)

        matched_detection_indices, unmatched_detection_indices = [], []
        for i in range(len(new_face_bboxes)):
            # matched detection
            if i in row_ind:
                j = col_ind[np.where(row_ind == i)[0][0]]
                matched_detection_indices.append([i, j])

            # unmatched detection
            else:
                unmatched_detection_indices.append(i)

        humans = []

        for inds in matched_detection_indices:
            i, j = inds
            human = self.humans[j]
            human.face_bounding_box_path.append(new_face_bboxes[i])
            human.facial_landmarks_path.append(new_facial_landmarks[i])
            humans.append(human)

        self.humans = humans

    def visualize(self, image):

        if self.regime == 'recognition':
            color = (255, 0, 0)
        elif self.regime == 'detection':
            color = (0, 0, 255)
        elif self.regime == 'tracking':
            color = (0, 255, 0)

        color = (255, 0, 0)


        for human in self.humans:

            id = human.person_id

            xmin, ymin, xmax, ymax = human.face_bounding_box_path[-1].bounding_box
            score = human.face_bounding_box_path[-1].confidence

            cv2.putText(image, id, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, color)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1)
            #cv2.putText(image, str(round(score, 2)), (xmax, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, color)

            landmarks, confidences = \
                    human.facial_landmarks_path[-1].landmarks, \
                    human.facial_landmarks_path[-1].confidence
            for j in range(68):
                x, y = landmarks[j]
                cv2.circle(image, (int(x), int(y)), 1, color, -1)
