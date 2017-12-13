# Many thanks to Daniel Stang for his great guide
# https://medium.com/@WuStangDan/step-by-step-tensorflow-object-detection-api-tutorial-part-1-selecting-a-model-a02b6aabe39e
# The following code was created using the above guide and the tensorflow example notebook
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

from styx_msgs.msg import TrafficLightState
import os
import numpy as np
import tensorflow as tf
from utils import label_map_util

class TLClassifier(object):
    def __init__(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        model = current_dir + '/ssd_inception_v2_coco/frozen_inference_graph.pb'
        labels_file = current_dir + '/label_map.pbtxt'
        num_classes = 4

        label_map = label_map_util.load_labelmap(labels_file)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes,
                                                                    use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            # Works up to here.
            with tf.gfile.GFile(model, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.d_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.d_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.d_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_d = self.detection_graph.get_tensor_by_name('num_detections:0')
            self.sess = tf.Session(graph=self.detection_graph)

        print("Loaded frozen model graph")

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        image_expanded = np.expand_dims(image, axis=0)
        with self.detection_graph.as_default():
            (boxes, scores, classes, num) = self.sess.run(
                [self.d_boxes, self.d_scores,
                 self.d_classes, self.num_d],
                feed_dict={self.image_tensor: image_expanded})

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        light_state = TrafficLightState.UNKNOWN

        score_threshold = .50
        for i in range(boxes.shape[0]):
            if scores is None or scores[i] > score_threshold:

                class_name = self.category_index[classes[i]]['name']

                if class_name == 'Red':
                    light_state = TrafficLightState.RED
                elif class_name == 'Green':
                    light_state = TrafficLightState.GREEN
                elif class_name == 'Yellow':
                    light_state = TrafficLightState.YELLOW

        return light_state
