from __future__ import print_function
from flask import Flask
import logging
import json
from flask import jsonify, request
from datetime import datetime
import sys
import base64
import StringIO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from object_detection import methods

dg = methods.build_graph(methods.model_path)
labels = methods.load_labels(methods.labels_path, methods.num_classes)
app = Flask(__name__)

@app.route('/')
def homepage():

    the_time = datetime.now().strftime("%A, %d %b %Y %l:%M %p")

    return """
    <h1>Hello heroku</h1>
    <p>It is currently {time}.</p>

    <img src="http://loremflickr.com/600/400">
    """.format(time=the_time)


def decode_img(img_str):
    img_bytes = base64.b64decode(img_str)
    tempBuff = StringIO.StringIO()
    tempBuff.write(img_bytes)
    tempBuff.seek(0) #need to jump back to the beginning before handing it off to PIL
    img = Image.open(tempBuff)
    return img

def img_to_np_arr(img):
    img_resp = np.array(img.getdata()).reshape(\
            (img.size[1], img.size[0], 3)).astype(np.uint8)
    img_resp = np.expand_dims(img_resp, axis=0)
    return img_resp


def box_drawer(draw, coords, width=1):
    for i in xrange(width):
        draw.rectangle(coords, outline='red')
        coords = (coords[0]+1,coords[1]+1,coords[2]+1,coords[3]+1) 
    return draw

def draw_boxes(img, predictions):
    draw = ImageDraw.Draw(img)
    img_width, img_height = img.size

    for prediction in predictions:
        box = prediction['box']
        ymin = int(img_width * box[0])
        xmin = int(img_width * box[1])
        ymax = int(img_width * box[2])
        xmax = int(img_width * box[3])
        coords = (xmin, ymin, xmax, ymax)
        draw = box_drawer(draw, coords, 10)
        draw.text((xmin, ymin), prediction['pred_class_name']['name'],\
                font=ImageFont.load_default())
    output_fp = StringIO.StringIO()
    img.save(output_fp,"png", quality=100)
    output_fp.seek(0)
    b64 = base64.b64encode(output_fp.getvalue())
    return b64




@app.route('/predict', methods=['POST'])
def test():
    data = request.get_json(force=True)
    try:
        img_str = data['img']
    except KeyError as e:
        return jsonify("'img' key not found in json")
    img = decode_img(img_str) 

    np_img = img_to_np_arr(img)
    
    predictions = methods.predict_image(np_img, dg, labels)

    draw_img = draw_boxes(img, predictions)

    output = dict(predictions=predictions, img=draw_img)
    return jsonify(output)




if __name__ == '__main__':
    app.run(debug=True)
