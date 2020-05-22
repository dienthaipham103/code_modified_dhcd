import sys
# sys.path.append('../insightface/deployment')
# sys.path.append('../insightface/src/common')

from imutils import paths
# import face_preprocess
import numpy as np
# import face_model
import argparse
import pickle
import cv2
import os
import intelligo_face
from database import insert

# model init
# model = insightface.model_zoo.get_model('retinaface_r50_v1')
# model.prepare(ctx_id = -1, nms=0.4)

# model_features = insightface.model_zoo.get_model('arcface_r100_v1')
# model_features.prepare(ctx_id = -1)

# model_faceanalysis = insightface.app.FaceAnalysis()
# ctx_id = -1
# model_faceanalysis.prepare(ctx_id = ctx_id, nms=0.4)

# Load the classifier model
# model_classifier = load_model('./models/classifier_model.h5')

# Load embeddings and labels
embedding_file = './models/embeddings.pickle'
label_file = './models/le.pickle'

ap = argparse.ArgumentParser()

ap.add_argument("--dataset", default="../datasets/face-train",
                help="Path to training dataset")
ap.add_argument("--embeddings", default="./models/embeddings.pickle")
ap.add_argument("--key", default="18FEAFE9E3CC0459299B6304B506E3CF",
                help="API access key")
# Argument of insightface
ap.add_argument('--image-size', default='112,112', help='')
ap.add_argument('--model', default='../insightface/models/model-y1-test2/model,0', help='path to load model.')
ap.add_argument('--ga-model', default='', help='path to load model.')
ap.add_argument('--gpu', default=0, type=int, help='gpu id')
ap.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
ap.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
ap.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')

args = ap.parse_args()

# Grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args.dataset))

# Initialize the faces embedder
# embedding_model = face_model.FaceModel(args)

model_faceanalysis = intelligo_face.app.FaceAnalysis()
model_faceanalysis.prepare(ctx_id=-1, le=None, classifier=None, embeddings=None, labels=None, logger=None, nms=0.4)

# Initialize our lists of extracted facial embeddings and corresponding people names
knownEmbeddings = []
knownNames = []

# Initialize the total number of faces processed
total = 0

# image_size
image_size = (112, 112)

# Loop over the imagePaths
for (i, imagePath) in enumerate(imagePaths):
    # extract the person name from the image path
    print(imagePath)
    print("[INFO] processing image {}/{}".format(i+1, len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]

    # load the image
    image = cv2.imread(imagePath)
    # convert face to RGB color
    nimg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # nimg = np.transpose(nimg, (2,0,1))
    # Get the face embedding vector

    print(nimg.shape)
    if nimg.shape[0:2] != image_size:
        # resize
        nimg = cv2.resize(nimg, image_size, interpolation=cv2.INTER_AREA)

        # test code
        # cv2.imshow('image', nimg)
        # cv2.waitKey(0)

    face_embedding = model_faceanalysis.get_embedding(nimg)
    print(face_embedding.shape)

    # test code
    print(face_embedding)

    # add the name of the person + corresponding face
    # embedding to their respective list
    knownNames.append(name)
    knownEmbeddings.append(face_embedding)
    total += 1

print(total, " faces embedded")

# test code
print(knownNames)
print('----------')
print(knownEmbeddings)

# save to output
# data = {"embeddings": knownEmbeddings, "names": knownNames}
# f = open(args.embeddings, "wb")
# f.write(pickle.dumps(data))
# f.close()

# insert into database
insert(knownNames, knownEmbeddings, args.key)
