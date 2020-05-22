#!/usr/bin/env python
# -*- coding: utf-8 -*-

import configparser
import logging
import os
import random
import string
import time
import boto
import werkzeug
from flask import Flask, send_from_directory
from flask_cors import CORS
from flask_restful import reqparse, abort, Resource, Api
from keras.models import load_model
from werkzeug.utils import secure_filename
import argparse
import pickle
import intelligo_face
import cv2
import numpy as np
from imutils import paths
import requests

np.set_printoptions(precision=2)

# *****************************
# *****************************
# add url of query engine here

WORKER_ENGINE_LIST = os.getenv('WORKER_ENGINE_LIST', 'http://127.0.0.1:8000').split(';')

# *****************************
MAX_INPUT_SIZE = os.getenv('FULL_IMAGE_SIZE', 1000)
UPLOAD_FOLDER = os.getenv('UPLOAD_FILE_DIR', './uploads')
LOG_DIR = os.getenv('LOG_DIR', './logs')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
NEW_IMAGE_FOLDER = os.getenv('NEW_IMAGE_FOLDER', './images/face-train')
LOGGING_LEVEL = os.getenv('LOGGING_LEVEL', 'INFO')

logfile_name = secure_filename(str(time.ctime() + '.logs'))
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(filename=LOG_DIR + '/' + logfile_name)

Config = configparser.RawConfigParser()
Config.read('./config/config.cfg')

AWS_ACCESS_KEY_ID = Config.get('AWSSettings', 'AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = Config.get('AWSSettings', 'AWS_SECRET_ACCESS_KEY')
bucketname = Config.get('AWSSettings', 'BUCKET_NAME')

# MD5 of facedemo.api.intelligo.xyz, bentrehospital.intelligo.xyz
API_ACCESS_KEY = ['18FEAFE9E3CC0459299B6304B506E3CF', 'C240CC161714C79158E3391ABAFCDF81']

args_glob = ""
arg_parser = ""

# Load embeddings and labels
classifier_file = './models/classifier_model.h5'
embedding_file = './models/embeddings.pickle'
label_file = './models/le.pickle'

model = intelligo_face.app.FaceAnalysis()

image_size = (112, 112)


def init_model():
    global model

    model_classifier = load_model(classifier_file)
    data = pickle.loads(open(embedding_file, "rb").read())
    le = pickle.loads(open(label_file, "rb").read())

    embeddings = np.array(data['embeddings'])
    labels = le.fit_transform(data['names'])

    model.prepare(ctx_id=-1, classifier=model_classifier, labels=labels,
                  embeddings=embeddings, le=le, logger=logger, nms=0.4,
                  max_img_size=int(MAX_INPUT_SIZE))


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# def random_string(length):
#    return ''.join(random.choice(string.ascii_lowercase) for i in range(length))


# def download_files(bucket_name, image_name_input, image_name_output):
#     logger.info('[REST-API] Downloading files ...')
#     # connect to the bucket
#     conn = boto.connect_s3(AWS_ACCESS_KEY_ID,
#                            AWS_SECRET_ACCESS_KEY)
#     logger.info('[REST-API] Access granted to bucket')
#     bucket = conn.get_bucket(bucket_name)
#     # go through the list of files
#     bucket_list = bucket.list()
#
#     for bucket_file in bucket_list:
#         file_path = str(bucket_file.key)
#         if file_path.endswith(image_name_input):
#             try:
#                 bucket_file.get_contents_to_filename(image_name_output)
#
#             except Exception as e:
#                 logger.info("[Face-Recognition] Trouble in saving file: {0}".format(e))
#
#     logger.info('[REST-API] Files downloaded')


# def upload_files(bucket_name, image_name_input, image_name_output):
#     logger.info('[REST-API] Uploading files ...')
#
#     # connect to the bucket
#     conn = boto.connect_s3(AWS_ACCESS_KEY_ID,
#                            AWS_SECRET_ACCESS_KEY)
#
#     logger.info('[REST-API] Accessing S3 bucket...')
#     bucket = conn.get_bucket(bucket_name)
#
#     k = bucket.new_key(image_name_output)
#     k.set_contents_from_filename(image_name_input)
#
#     logger.info('[REST-API] Files uploaded')


# def upload_folder(bucket_name, image_folder, key):
#     logger.info('[REST-API] Uploading <{}> folder ...'.format(image_folder))
#
#     # connect to the bucket
#     conn = boto.connect_s3(AWS_ACCESS_KEY_ID,
#                            AWS_SECRET_ACCESS_KEY)
#
#     logger.info('[REST-API] Accessing S3 bucket...')
#     bucket = conn.get_bucket(bucket_name)
#
#     imgPaths = paths.list_images(image_folder)
#     for img in imgPaths:
#         s3_key = '/'.join(img.split(os.path.sep)[-2:])
#         s3_key = 'train-datasets/' + key + '/' + s3_key
#         logger.info("Uploading <{}> to <{}>".format(img, s3_key))
#         k = bucket.new_key(s3_key)
#         k.set_contents_from_filename(img)
#
#     logger.info('[REST-API] Folder <{}> uploaded'.format(image_folder))


def update_model(model_file_name, key):
    os.system('python ./faces_embedding_512.py --dataset={} --key={}'.format(NEW_IMAGE_FOLDER, key))
    os.system('python ./train_softmax.py --model={} --key={}'.format(model_file_name, key))

    # after training, remove all temporal data
    model.clear_buffer()
    # upload temporal data to S3
    try:
        # upload_folder('facerecognition-demo', NEW_IMAGE_FOLDER, key)---------------no need----------------

        # delete uploaded images
        for root, dirs, files in os.walk(NEW_IMAGE_FOLDER, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))

        # notify other query engines ???????????????????
        for host_name in WORKER_ENGINE_LIST:
            logger.info('[REST_API] Notify ' + host_name + '/notify/model')
            requests.get(host_name + '/notify/model', headers={'x-access-key': API_ACCESS_KEY[0]})

    except Exception as e:
        print(e)


# def add_face(name, img_file):
#     logger.info("[REST-API] Add Face Processing === {} ===".format(img_file))
#     image = cv2.imread(img_file)
#     # convert face to RGB color
#     nimg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     # Get the face embedding vector
#     if nimg.shape[0:2] != image_size:
#         nimg = cv2.resize(nimg, image_size, interpolation=cv2.INTER_AREA)
#
#     face_embedding = model.get_embedding(nimg)
#     # this embedding and name will be added to database when we call retrain
#     # here, we temporally store them in memory to query
#     model.add_buffer(face_embedding, name)
#
#     # notify other query engines here
#     for host_name in WORKER_ENGINE_LIST:
#         data = {'embedding': face_embedding.tolist()}
#         requests.post(url=host_name + '/notify/face/' + name, json=data, headers={'x-access-key': API_ACCESS_KEY[0]})


class RetrainModel(Resource):
    def post(self):
        logger.info('[REST-API] Received a POST Call for retrain model')
        args = parser.parse_args()
        key = args['x-access-key']

        if key in API_ACCESS_KEY:
            try:
                logger.info('[REST-API] Start to retrain model!')
                update_model(classifier_file, key)
                logger.info('[REST-API] reload new model')
                init_model()
            except:
                logger.error('[REST-API] FaceRecognition retrain model engine error')
                abort(500, message='Preprocess failed')

            api_response = {
                "status": "success"
            }
            return api_response, 201
        else:
            api_response = {
                "status": "x-access-key is not correct!"
            }
            return api_response, 401

    def get(self):
        return 'ok'


class FaceQuery(Resource):
    def post(self):
        logger.info('[REST-API] Received a POST Call for face query')

        args = parser.parse_args()
        imgfile = args['file']
        key = args['x-access-key']
        multi = args['multi-face']

        if key in API_ACCESS_KEY and not(imgfile is None):
            if imgfile.filename == '' or not allowed_file(imgfile.filename):
                return {'status': 'No file name or file type is not in [png, jpg, jpeg]!'}, 401

            path_to_image = secure_filename(str(time.ctime()) + ' ' + imgfile.filename)
            path_to_image = os.path.join(UPLOAD_FOLDER, path_to_image)
            imgfile.save(path_to_image)

            try:
                multi_flag = False
                if multi:
                    multi_flag = True

                results = model.predict(image_file=path_to_image, multiple=multi_flag)
            except:
                logger.error('[REST-API] Face Recognition engine error')
                abort(500, message='Preprocess failed')

            logger.info('[REST-API] Image process is completed!')

            os.remove(path_to_image)

            logger.info('[REST-API] Preparing the output!')

            data = []
            for face_name, confidence in results.items():
                data.append({
                    'face': face_name,
                    'confidence': str(confidence)
                })

            logger.info('[REST-API] Done with the output data!')
            api_response = {
                "faces": data
            }
            return api_response, 201

        else:
            api_response = {
                "status": "x-access-key is not correct or missing attachment !"
            }
            return api_response, 401

    def get(self):
        return 'ok'


# class FaceAdd(Resource):
#     def post(self, face_id):
#         logger.info('[REST-API] Received a POST Call for adding face_id <{}>'.format(face_id))
#         args = parser.parse_args()
#         imgfile = args['file']
#         key = args['x-access-key']
#         multi = args['multi-face']
#
#         if key in API_ACCESS_KEY and not imgfile is None:
#             if imgfile.filename == '' or not allowed_file(imgfile.filename):
#                 return {'status': 'No file name or file type is not in [png, jpg, jpeg]!'}, 401
#
#             if multi:
#                 path_to_image_in = secure_filename(str(time.ctime()) + ' temp ' + imgfile.filename)
#                 path_to_folder = os.path.join(NEW_IMAGE_FOLDER, face_id)
#                 if not os.path.exists(path_to_folder):
#                     os.makedirs(path_to_folder)
#                 path_to_image_in = os.path.join(path_to_folder, path_to_image_in)
#                 imgfile.save(path_to_image_in)
#
#                 path_to_image = secure_filename(str(time.ctime()) + ' ' + imgfile.filename)
#                 path_to_image = os.path.join(path_to_folder, path_to_image)
#                 # crop face
#                 model.crop(path_to_image_in, path_to_image)
#                 os.remove(path_to_image_in)
#             else:
#                 path_to_image = secure_filename(str(time.ctime()) + ' ' + imgfile.filename)
#                 path_to_folder = os.path.join(NEW_IMAGE_FOLDER, face_id)
#                 if not os.path.exists(path_to_folder):
#                     os.makedirs(path_to_folder)
#                 path_to_image = os.path.join(path_to_folder, path_to_image)
#
#                 imgfile.save(path_to_image)
#
#             add_face(face_id, path_to_image)
#             api_response = {
#                 "status": "success"
#             }
#             return api_response, 201
#         else:
#             api_response = {
#                 "status": "x-access-key is not correct or missing attachment!"
#             }
#             return api_response, 401
#
#     def get(self):
#         return 'ok'


class FacePosition(Resource):
    def post(self):
        logger.info('[REST-API] Received a face lookup...')
        args = parser.parse_args()
        imgfile = args['file']
        key = args['x-access-key']

        if key in API_ACCESS_KEY and not imgfile is None:
            if imgfile.filename == '' or not allowed_file(imgfile.filename):
                return {'status': 'No file name or file type is not in [png, jpg, jpeg]!'}, 401

            path_to_image_in = secure_filename(str(time.ctime()) + ' temp ' + imgfile.filename)
            path_to_image_in = os.path.join(UPLOAD_FOLDER, path_to_image_in)
            imgfile.save(path_to_image_in)

            # face lookup
            pos = model.position(path_to_image_in)
            os.remove(path_to_image_in)

            api_response = {
                "position": pos
            }
            return api_response, 201
        else:
            api_response = {
                "status": "x-access-key is not correct or missing attachment !"
            }
            return api_response, 401

    def get(self):
        return 'ok'


class Download(Resource):
    def get(self, filename):
        logger.info('[REST-API] Received a download: ' + filename)
        return send_from_directory(os.path.realpath('./models'), filename)


def init_arg():

    arg_parser = argparse.ArgumentParser(conflict_handler='resolve')

    arg_parser.add_argument('--imgs', type=str, nargs='+', default="./images/examples/carell.jpg",
                             help="Input image.")
    arg_parser.add_argument('--multi', default=False, help="Infer multiple faces in image",
                             action="store_true")
    arg_parser.add_argument('--port', type=int, default=9000, help="Listen port")

    global args_glob
    args_glob = arg_parser.parse_args()


if __name__ == "__main__":

    init_arg()

    init_model()

    # API Definition
    app = Flask(__name__)
    CORS(app)
    api = Api(app)

    parser = reqparse.RequestParser()
    parser.add_argument('file', type=werkzeug.datastructures.FileStorage, location='files')
    parser.add_argument('x-access-key', type=str, required=True, location='headers')
    parser.add_argument('multi-face', type=bool, location='headers')

    api.add_resource(RetrainModel, '/admin/retrain')
    api.add_resource(Download, '/admin/download/<string:filename>')
    # api.add_resource(FaceAdd, '/admin/face-add/<string:face_id>')
    api.add_resource(FaceQuery, '/query/face-id')
    api.add_resource(FacePosition, '/query/face-position')

    # Launch the app
    app.run(host='0.0.0.0', port=args_glob.port, threaded=False)
