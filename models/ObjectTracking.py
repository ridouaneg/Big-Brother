import cv2

from models.ObjectDetection import ObjectDetectorResult, BoundingBox

class ObjectTracker:

    def __init__(self, model_path='CSRT', do_timing=False):
        self.set_model(model_path)
        self.do_timing = do_timing

    def set_model(self, tracker_type):
        if tracker_type == 'BOOSTING':
            self.model = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            self.model = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            self.model = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            self.model = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            self.model = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            self.model = cv2.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            self.model = cv2.TrackerMOSSE_create()
        if tracker_type == 'CSRT':
            self.model = cv2.TrackerCSRT_create()

    def initialize(self, image, bbox):
        _ = self.model.init(image, bbox)

    def update(self, image):
        _, tmp = self.model.update(image)
        bbox = [int(tmp[0]), int(tmp[1]), int(tmp[0] + tmp[2]), int(tmp[1] + tmp[3])]
        return bbox

class MultiObjectTracker:

    def __init__(self, model_path='CSRT', do_timing=False):
        self.model = model_path
        self.do_timing = do_timing
        self.tracker = []

    def initialize(self, image, bboxes):
        self.trackers = [ObjectTracker(model_path=self.model, do_timing=self.do_timing) for bbox in bboxes]
        for tracker, bbox in zip(self.trackers, bboxes):
            tracker.initialize(image, (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]))

    def update(self, image):
        bboxes = [tracker.update(image) for tracker in self.trackers]
        scores = [1. for tracker in self.trackers]
        self.result = ObjectDetectorResult([BoundingBox(bbox, score) for bbox, score in zip(bboxes, scores)])
        return bboxes, scores

    def get_result(self):
        return self.result
