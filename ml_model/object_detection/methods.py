#!/usr/bin/env python


import numpy as np
import os
from os.path import join
import tensorflow as tf
from PIL import Image

from utils import label_map_util
import json

model_path = "model_data/frozen_inference_graph.pb"
labels_path = "model_data/mscoco_label_map.pbtxt"

'''
model_path = "model_data/bigger_inference_graph.pb"
labels_path = "model_data/bigger_label_map.pbtxt"
'''

dirpath = os.path.dirname(os.path.abspath(__file__))
model_path = join(dirpath, model_path)
labels_path = join(dirpath, labels_path)


num_classes = 90

def build_graph(model_path):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(model_path, 'rb') as fp:
            serialized_graph = fp.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph


def load_labels(labels_path, num_classes):
    label_map = label_map_util.load_labelmap(labels_path)
    categories = label_map_util.convert_label_map_to_categories(\
            label_map, max_num_classes=num_classes, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return category_index

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


def run_model(img, dg):
    with dg.as_default():
        with tf.Session(graph=dg) as sess:
            image_tensor = dg.get_tensor_by_name('image_tensor:0')
            detection_boxes = dg.get_tensor_by_name('detection_boxes:0')
            detection_scores = dg.get_tensor_by_name('detection_scores:0')
            detection_classes = dg.get_tensor_by_name('detection_classes:0')
            num_detections = dg.get_tensor_by_name('num_detections:0')

            boxes, scores, classes, num = sess.run(\
                    [detection_boxes, detection_scores, detection_classes,\
                    num_detections],\
                    feed_dict={image_tensor:img})
            return boxes, scores, classes, num

def predict_image(img, dg, labels):
    boxes, scores, classes, num = run_model(img, dg)
    scores = scores[0].tolist()
    idx_to_keep = []

    for i in xrange(len(scores)):
        score = scores[i]
        if score > 0.5:
            idx_to_keep.append(i)

    predictions = []
    for idx in idx_to_keep:
        box = boxes[0][idx].tolist()
        score = float(scores[idx])
        pred_class = int(classes[0][idx])
        pred_class_name = labels[pred_class]
        predictions.append(dict(box=box, score=score, pred_class=pred_class,\
                pred_class_name=pred_class_name))
    return predictions


'''

dg = build_graph(model_path)
labels = load_labels(labels_path, num_classes)


img_dir = "./test_images"
img_paths = [join(img_dir, x) for x in os.listdir(img_dir)]
for img_path in img_paths:
    print img_path
    predictions = predict_image(img_path, dg, labels)
    print json.dumps(predictions, indent=4)
    raw_input()


'''
