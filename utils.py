# Imports
import numpy as np
import cv2
import dlib

def load_image(path):
    # Given the path to an image, load it into a numpy array
    # input : the path to an jpg/png image
    # output : a numpy array of the image
    image = cv2.imread(path)
    return np.array(image)

def rect_to_list(rect):
    # Convert a dlib rectangle into a list of coordinates
    # input : a dlib rectangle
    # output : a list of int coordinates [left, top, right, bottom]
    return [rect.left(), rect.top(), rect.right(), rect.bottom()]

def list_to_rect(l):
    # Convert a list of coordinates into a dlib rectangle
    # input : a list of int coordinates [left, top, right, bottom]
    # output : a dlib rectangle
    return dlib.rectangle(l[0], l[1], l[2], l[3])

def face_encodings_distance(face_encoding1, face_encoding2):
    # Given two face descriptors, return the distance between them
    # input : two descriptors to compare (two 128-dim vectors)
    # output : the distance between those descriptors (the L1-norm)
    return np.linalg.norm(face_encoding1 - face_encoding2)

def IoU(bndbx_d, bndbx_l):
    xmin1, ymin1, xmax1, ymax1 = bndbx_d[0], bndbx_d[1], bndbx_d[2], bndbx_d[3]
    xmin2, ymin2, xmax2, ymax2 = bndbx_l[0], bndbx_l[1], bndbx_l[2], bndbx_l[3]

    xA = max(xmin1, xmin2)
    yA = max(ymin1, ymin2)
    xB = min(xmax1, xmax2)
    yB = min(ymax1, ymax2)
    intersection = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    aire1 = (xmax1 - xmin1 + 1) * (ymax1 - ymin1 + 1)
    aire2 = (xmax2 - xmin2 + 1) * (ymax2 - ymin2 + 1)
    union = aire1 + aire2 - intersection

    eps = 10**(-8)
    iou = intersection / (union + eps)
    return iou

def global_distances(l1, l2):
    n1, n2 = len(l1), len(l2)
    distances = [[IoU(l1[i], l2[j]) for j in range(n2)] for i in range(n1)]
    return distances

def draw_on_frame(frame, faces, names):
    # Given a frame and two lists of face coordinates and names,
    # draw rectangles and write names on the frame
    # Input : a frame (numpy array), faces coordinates (list) and names (list)
    # Output : nothing
    for face, name in zip(faces, names):
        left, top, right, bottom = face[0], face[1], face[2], face[3]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
