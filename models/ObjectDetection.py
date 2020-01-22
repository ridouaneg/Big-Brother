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
    bounding_box : numpy array of shape (4, 1) and dtype float, [xmin, ymin, xmax, ymax]
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
    of type 'BoungingBox'.


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
        'BoundingBox' objects.

        Returns
        ------
        bounding_boxes : list of numpy array of shape (4, 1) and type float
        scores : list of float
        """
        bounding_boxes = [bbox.bounding_box for bbox in self.bounding_boxes]
        confidences = [bbox.confidence for bbox in self.bounding_boxes]
        return bounding_boxes, confidences

class ObjectDetector:
    """
    This class contains the object detection model.

    When we have decteted several objects, we store them into a list of elements
    of type 'BoungingBox'.


    Attributes
    ----------
    bounding_boxes : list of 'BoundingBox' objects

    Methods
    -------
    get_result()
        Return the result of the object detector.
    predict(image)
        Predict object bounding boxes and classes from a RGB image.
    visualize(image, bounding_boxes=None, confidences=None, color=(255, 0, 0))
        Draw detected bounding boxes on the image and display confidence scores.
    """

    def __init__(self, threshold=0.25, input_size=(512, 1024, 3), do_timing=False):
        self.threshold = threshold
        self.input_size = input_size
        self.image_size = None
        self.do_timing = do_timing
        self.result = None

    def get_result(self):
        """Return the result of the object detector."""
        return self.result

    def predict(self, image):
        """Predict object bounding boxes and classes from a RGB image.

        Parameters
        ----------
        image : numpy array of shape (height, width, 3)
            The input image
        """

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
        """Draw detected bounding boxes on the image and display confidence scores.

        Parameters
        ----------
        image : numpy array of shape (height, width, 3)
            The image we want to draw on
        bounding_boxes : [(xmin, ymin, xmax, ymax), ... ], list of 4-tuple of floats
            The bounding boxes coordinates of detected humans on the image
        confidences : list of float
            The corresponding confidence for each bounding box
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
    """
    This class contains the face detection model.

    Attributes
    ----------
    model_path : str
        the path/name of the model
        models available :
            'HOG' -> no deep learning, fast but inaccurate
            'MTCNN' -> deep learning, slow but accurate
    threshold : float
        the detection threshold
    input_size : (height, width, channels), 3-tuple of int
        the input size of the model
    do_timing : bool
        if True, display the runtime of the detection pipeline

    Methods
    -------
    set_model(model_path)
        Initialize the model from its name/path
    preprocess(image)
        Pre-process the image before feeding it into the model
    model_predict(model_input)
        Predict human bounding boxes from a RGB image
    post_process(model_output)
        Postprocess the model output
    """

    def __init__(self, model_path='HOG', threshold=0.50, input_size=(512, 1024, 3), do_timing=False):
        super().__init__(threshold, input_size, do_timing)
        self.model_path = model_path
        self.set_model(model_path)

    def set_model(self, model_path):
        """Initialize the model from its name/path."""
        if model_path == 'HOG':
            from dlib import get_frontal_face_detector
            self.model = get_frontal_face_detector()
        elif model_path == 'MTCNN':
            from mtcnn.mtcnn import MTCNN
            self.model = MTCNN()

    def pre_process(self, image):
        """Pre-process the image before feeding it into the model."""
        # save image size
        self.image_size = np.shape(image)
        # resize
        #model_input = cv2.resize(image, (self.input_size[1], self.input_size[0]))
        model_input = image
        return model_input

    def model_predict(self, model_input):
        """Predict face bounding boxes from a RGB image."""
        model_output = []

        if self.model_path == 'HOG':
            model_output = self.model(model_input)

        elif self.model_path == 'MTCNN':
            model_output = self.model.detect_faces(model_input)

        return model_output

    def post_process(self, model_output):
        """Postprocess the model output."""
        bounding_boxes, confidences = [], []

        if self.model_path == 'HOG':
            bounding_boxes, confidences = [], []
            for f in model_output:
                xmin, ymin, xmax, ymax = Util.rect_to_list(f)

                confidence = 1.00

                bounding_boxes.append([int(xmin), int(ymin), int(xmax), int(ymax)])
                confidences.append(confidence)

        elif self.model_path == 'MTCNN':
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
