import base64
import packages.tfObjWebrtc.object_detection_api_v1 as object_detection_api_v1
import packages.tfObjWebrtc.object_detection_api_v3 as object_detection_api_v3
import cv2
import numpy as np

from flask import Flask, request, jsonify
from face_recognition import detect_face, get_face_rectangle
from PIL import Image
from flask_cors import cross_origin, CORS

app = Flask('Object_Recognition')
CORS(app, supports_credentials=True)


@app.route('/face_recognition', methods=['POST'])
def face_recognition():
    # face-api
    image_file_str = request.values.get('image')
    b64str = image_file_str.split(',')[1]
    image = np.frombuffer(base64.b64decode(b64str), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    det, detected_image = detect_face(image)
    detected_image = cv2.imencode('.png', detected_image)[1]
    str_b64img = base64.b64encode(detected_image).decode()

    res = {
        "det": det,
        "img_stream": str_b64img
    }
    return jsonify(res)


@app.route('/object_detection_v1', methods=['POST'])
def object_detection_v1():
    # object-api
    try:
        # image_file = request.files['image']  # get the image
        image_file_str = request.values.get('image')
        threshold = request.values.get('threshold')
        b64str = image_file_str.split(',')[1]
        image = np.frombuffer(base64.b64decode(b64str), np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # Set an image confidence threshold value to limit returned data
        # threshold = request.form.get('threshold')
        # if threshold is None:
        #     threshold = 0.5
        # else:
        #     threshold = float(threshold)

        # finally run the image through tensor flow object detection`
        # image_object = Image.open(image_file)
        image_shape = image.shape
        objects = object_detection_api_v1.run_inference_for_single_image(object_detection_api_v1.detection_model,
                                                                         image)
        faces, _ = detect_face(image)
        faces_rectangle = get_face_rectangle(faces, image_shape)
        res = {
            'objects': objects,
            'success': True,
        }
        return jsonify(res)

    except Exception as e:
        print('POST /image error: %e' % e)
        return e


@app.route('/object_detection_v3', methods=['POST'])
def object_detection_v3():
    # object-api
    try:
        image_file_str = request.values.get('image')
        threshold = request.values.get('threshold')
        b64str = image_file_str.split(',')[1]
        image = np.frombuffer(base64.b64decode(b64str), np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        objects = object_detection_api_v3.get_objects(image)
        res = {
            'objects': objects,
            'success': True,
        }
        return jsonify(res)

    except Exception as e:
        print('POST /image error: %e' % e)
        return e


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
