import numpy as np
import cv2
import time

from Util import Util

class BoundingBox:
    """
    This class contains information of an object bounding box.

    For each detected objects, we store the corresponding bounding box into this
    class. We can then access the bounding box coordinates, confidence and a
    dictionnary which contains useful information.


    Attributes
    ----------
    bounding_box : numpy array of shape (4, 1) and type float, [xmin, ymin, xmax, ymax]
    confidence : float
    bounding_box_dict : dict
    """

    def __init__(self, bounding_box, confidence):
        self.bounding_box = bounding_box
        self.confidence = confidence
        self.bounding_box_dict = {
            'xmin':bounding_box[0],
            'ymin':bounding_box[1],
            'xmax':bounding_box[2],
            'ymax':bounding_box[3],
            'w':bounding_box[2] - bounding_box[0],
            'h':bounding_box[3] - bounding_box[1],
            'confidence':confidence
        }

class ObjectDetectorResult:
    """
    This class contains information of the result of an object detector.

    When we have decteted several objects, we store them into a list of elements
    of type 'BoungingBox'


    Attributes
    ----------
    bounding_boxes : list of 'BoundingBox' objects

    Methods
    -------
    convert_to_list()
        Extract bounding boxes coordinates and confidences from the list of
        'BoundingBox' objects
    """

    def __init__(self, bounding_boxes=[]):
        self.bounding_boxes = bounding_boxes

    def convert_to_list(self):
        """Extract bounding boxes coordinates and confidences from the list of
        'BoundingBox' objects

        Returns
        ------
        bounding_boxes : list of numpy array of shape (4, 1) and type float
        scores : list of float
        """
        bounding_boxes = [bbox.bounding_box for bbox in self.bounding_boxes]
        confidences = [bbox.confidence for bbox in self.bounding_boxes]
        return bounding_boxes, confidences

class ObjectDetector:

    def __init__(self, threshold=0.25, input_size=(512, 1024, 3), do_timing=False):
        self.threshold = threshold
        self.input_size = input_size
        self.image_size = None
        self.do_timing = do_timing
        self.result = None

    def get_result(self):
        return self.result

    def predict(self, image):

        # pre-process
        pre_process_runtime_start = time.time()
        model_input = self.pre_process(image)
        pre_process_runtime_end = time.time()

        # model prediction
        model_predict_runtime_start = time.time()
        model_output = self.model_predict(model_input)
        model_predict_runtime_end = time.time()

        # postprocess
        post_process_runtime_start = time.time()
        bounding_boxes, confidences = self.post_process(model_output)
        post_process_runtime_end = time.time()

        self.result = ObjectDetectorResult([BoundingBox(bounding_box, confidence) for bounding_box, confidence in zip(bounding_boxes, confidences)])

        if self.do_timing:
            print('object detector preprocessing time (ms):', (pre_process_runtime_end - pre_process_runtime_start) * 1e3)
            print('object detector prediction time (ms):', (model_predict_runtime_end - model_predict_runtime_start) * 1e3)
            print('object detector post-processing time (ms):', (post_process_runtime_end - post_process_runtime_start) * 1e3)

        return bounding_boxes, confidences

    def visualize(self, image, bounding_boxes=None, confidences=None, color=(255, 0, 0)):
        """Draw detected bounding boxes on the image

        Parameters
        ----------
        image : numpy array of shape (height, width, channels)
            The image
        """
        if bounding_boxes is None:
            bounding_boxes, confidences = self.result.convert_to_list()
        else:
            bounding_boxes, confidences = bounding_boxes, confidences

        for k in range(len(bounding_boxes)):

            xmin, ymin, xmax, ymax = bounding_boxes[k]
            confidence = round(confidences[k], 2)

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1)
            cv2.putText(image, str(confidence), (xmax, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, color)

class FaceDetector(ObjectDetector):

    def __init__(self, model_path='HOG', threshold=0.50, input_size=(512, 1024, 3), do_timing=False):
        super().__init__(threshold, input_size, do_timing)
        self.model_path = model_path
        self.set_model(model_path)

    def set_model(self, model_path):
        if model_path == 'HOG':
            from dlib import get_frontal_face_detector
            self.model = get_frontal_face_detector()
        elif model_path == 'MTCNN':
            from mtcnn.mtcnn import MTCNN
            self.model = MTCNN()

    def pre_process(self, image):
        # save image size
        self.image_size = np.shape(image)
        # resize
        #model_input = cv2.resize(image, (self.input_size[1], self.input_size[0]))
        model_input = image
        return model_input

    def model_predict(self, model_input):
        model_output = []

        if self.model_path == 'HOG':
            model_output = self.model(model_input)

        elif self.model_path == 'MTCNN':
            model_output = self.model.detect_faces(model_input)

        return model_output

    def post_process(self, model_output):

        bounding_boxes, confidences = [], []

        if self.model_path == 'HOG':
            bounding_boxes, confidences = [], []
            for f in model_output:
                xmin, ymin, xmax, ymax = Util.rect_to_list(f)

                confidence = 1.00

                bounding_boxes.append([int(xmin), int(ymin), int(xmax), int(ymax)])
                confidences.append(confidence)

        elif self.model_path == 'MTCNN':
            print(model_output)
            output_bounding_boxes = [model_output[i]['box'] for i in range(len(model_output))]
            output_confidences = [[model_output[i]['confidence']] for i in range(len(model_output))]

            bounding_boxes, confidences = [], []

            nb_dets = len(output_bounding_boxes)

            for i in range(nb_dets):

                confidence = output_confidences[i][0]

                if confidence < self.threshold:
                    continue

                xmin, ymin, w, h = output_bounding_boxes[i]
                xmax, ymax = xmin + w, ymin + h

                bounding_boxes.append([int(xmin), int(ymin), int(xmax), int(ymax)])
                confidences.append(confidence)

        return bounding_boxes, confidences
