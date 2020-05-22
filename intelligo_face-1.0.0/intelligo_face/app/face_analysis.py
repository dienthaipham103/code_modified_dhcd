from __future__ import division

import collections

import numpy as np
import os
import cv2
from ..model_base import model_base
from ..utils import face_align

__all__ = ['FaceAnalysis',
           'Face']

Face = collections.namedtuple('Face', [
        'bbox', 'landmark', 'det_score', 'embedding', 'gender', 'age', 'embedding_norm', 'normed_embedding'])

Face.__new__.__defaults__ = (None,) * len(Face._fields)

# Define distance function
def findCosineDistance(vector1, vector2):
    """
    Calculate cosine distance between two vector
    """
    vec1 = vector1.flatten()
    vec2 = vector2.flatten()

    a = np.dot(vec1.T, vec2)
    b = np.dot(vec1.T, vec1)
    c = np.dot(vec2.T, vec2)
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def CosineSimilarity(test_vec, source_vecs):
    """
    Verify the similarity of one vector to group vectors of one class
    """
    cos_dist = 0
    for source_vec in source_vecs:
        cos_dist += findCosineDistance(test_vec, source_vec)
    return cos_dist / len(source_vecs)


class FaceAnalysis:
    def __init__(self, det_name='retinaface_r50_v1', rec_name='arcface_r100_v1'):
        assert det_name is not None
        self.det_model = model_base.get_model(det_name)
        self.rec_model = model_base.get_model(rec_name)
        self.logger = None
        self.max_image_resize = None
        self.classifier_model = None
        self.le = None
        self.embeddings = None
        self.labels = None
        self.new_embeddings = []
        self.new_knownnames = []

    def prepare(self, ctx_id, classifier, le, embeddings, labels, logger, max_img_size = 1000, nms=0.4):
        self.det_model.prepare(ctx_id, nms)
        self.classifier_model = classifier
        self.logger = logger
        self.max_image_resize = max_img_size
        self.le = le
        self.embeddings = embeddings
        self.labels = labels
        self.rec_model.prepare(ctx_id)

    def add_buffer(self, embedding, knownname):
        self.new_embeddings.append(embedding)
        self.new_knownnames.append(knownname)

    def clear_buffer(self):
        self.new_embeddings.clear()
        self.new_knownnames.clear()

    def analyze(self, image, det_thresh = 0.8, det_scale = 1.0, max_num = 0):
        bboxes, landmarks = self.det_model.detect(image, threshold=det_thresh, scale = det_scale)
        if bboxes.shape[0]==0:
            return []
        if max_num>0 and bboxes.shape[0]>max_num:
            area = (bboxes[:,2]-bboxes[:,0])*(bboxes[:,3]-bboxes[:,1])
            img_center = image.shape[0] // 2, image.shape[1] // 2
            offsets = np.vstack([ (bboxes[:,0]+bboxes[:,2])/2-img_center[1], (bboxes[:,1]+bboxes[:,3])/2-img_center[0] ])
            offset_dist_squared = np.sum(np.power(offsets,2.0),0)
            values = area-offset_dist_squared*2.0 # some extra weight on the centering
            bindex = np.argsort(values)[::-1] # some extra weight on the centering
            bindex = bindex[0:max_num]
            bboxes = bboxes[bindex, :]
            landmarks = landmarks[bindex, :]
        ret = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i,4]
            landmark = landmarks[i]
            _img = face_align.norm_crop(image, landmark = landmark)
            # embedding = None
            embedding_norm = None
            normed_embedding = None
            gender = None
            age = None
            embedding = self.rec_model.get_embedding(_img).flatten()
            # embedding_norm = norm(embedding)
            # normed_embedding = embedding / embedding_norm
            # if self.ga_model is not None:
            #     gender, age = self.ga_model.get(_img)
            face = Face(bbox = bbox, landmark = landmark, det_score = det_score, embedding = embedding,
                        gender = gender, age = age, normed_embedding=normed_embedding,
                        embedding_norm = embedding_norm)
            ret.append(face)
        return ret

    def get_embedding(self, image):
        return self.rec_model.get_embedding(image)

    def position(self, img_file):
        self.logger.info('[Intelligo-Face] Face lookup from {} ...'.format(img_file))
        if not os.path.isfile(img_file):
            return None

        try:
            image = cv2.imread(img_file)
            height = image.shape[0]
            width = image.shape[1]
        except:
            return None

        resize_ratio = self.max_image_resize / max(width, height)
        # Calculate future image size
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        image = cv2.resize(image, target_size, fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
        boxes, landmarks = self.det_model.detect(image, threshold=0.5, scale=1.0)
        pos = []
        for i in range(boxes.shape[0]):
            box = boxes[i].astype(np.int) / resize_ratio
            crop_size = max(box[2] - box[0] + 1, box[3] - box[1] + 1)

            tl_x, tl_y = int((box[2] + box[0]) / 2 - crop_size / 2), int(
                (box[3] + box[1]) / 2 - crop_size / 2)
            br_x, br_y = int((box[2] + box[0]) / 2 + crop_size / 2), int(
                (box[3] + box[1]) / 2 + crop_size / 2)

            tl_y = max(0, tl_y)
            br_y = min(height, br_y)
            tl_x = max(0, tl_x)
            br_x = min(width, br_x)
            pos.append((tl_y, br_y, tl_x, br_x))
        return pos

    def crop(self, image_in, image_out, image_out_size=(112, 112)):
        self.logger.info('[Intelligo-Face] Crop face from {} ...'.format(image_in))
        if not os.path.isfile(image_in):
            return None

        try:
            image = cv2.imread(image_in)
            height = image.shape[0]
            width = image.shape[1]
        except:
            return None

        resize_ratio = self.max_image_resize / max(width, height)
        # Calculate future image size
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        image = cv2.resize(image, target_size, fx=0, fy=0, interpolation=cv2.INTER_CUBIC)

        boxes, landmarks = self.det_model.detect(image, threshold=0.5, scale=1.0)

        if boxes.shape[0] > 0:
            box = boxes[0].astype(np.int)
            crop_size = max(box[2] - box[0] + 1, box[3] - box[1] + 1)

            tl_x, tl_y = int((box[2] + box[0]) / 2 - crop_size / 2), int(
                (box[3] + box[1]) / 2 - crop_size / 2)
            br_x, br_y = int((box[2] + box[0]) / 2 + crop_size / 2), int(
                (box[3] + box[1]) / 2 + crop_size / 2)

            tl_y = max(0, tl_y)
            br_y = min(target_size[1], br_y)
            tl_x = max(0, tl_x)
            br_x = min(target_size[0], br_x)
            self.logger.info('[Intelligo-Face] Crop {}-{}, {}-{}'.format(tl_y, br_y, tl_x, br_x))
            crop_img = image[tl_y:br_y, tl_x:br_x]
            crop_img = cv2.resize(crop_img, image_out_size, fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(image_out, crop_img)
            self.logger.info('[Intelligo-Face] Crop face to {} successful'.format(image_out))
        return True

    def predict(self, image_file, multiple=False, image_size=(112, 112),
                cosine_threshold = 0.6,
                proba_threshold = 0.85,
                comparing_num = 5):
        results = {}

        self.logger.info("[Intelligo-Face] {} - multi: {}".format(image_file, multiple))

        if not os.path.isfile(image_file):
            return None

        try:
            img = cv2.imread(image_file)
        except:
            return None

        # client upload a large image with multiple faces
        if multiple:

            height = img.shape[0]
            width = img.shape[1]

            resize_ratio = self.max_image_resize / max(width, height)
            # Calculate future image size
            target_size = (int(resize_ratio * width), int(resize_ratio * height))
            self.logger.info("[Intelligo-Face] Scaled {} ".format(target_size))
            # Resize image
            resized_image = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            reps = self.analyze(resized_image)
            for idx, face in enumerate(reps):
                embedding = face.embedding
                embedding = np.reshape(embedding, (1, -1))
                # print(embedding.shape)
                # Predict class
                preds =  self.classifier_model.predict(embedding)
                preds = preds.flatten()
                # Get the highest accuracy embedded vector
                j = np.argmax(preds)
                proba = preds[j]
                # Compare this vector to source class vectors to verify it is actual belong to this class
                match_class_idx = (self.labels == j)
                match_class_idx = np.where(match_class_idx)[0]
                selected_idx = np.random.choice(match_class_idx, comparing_num)
                compare_embeddings = self.embeddings[selected_idx]
                # Calculate cosine similarity
                cos_similarity = CosineSimilarity(embedding, compare_embeddings)
                self.logger.info("[Intelligo-Face] dist {} -- prob {}".format(cos_similarity, proba))
                if cos_similarity < cosine_threshold and proba > proba_threshold:
                    name = self.le.classes_[j]
                    # text = "{}".format(name)
                    self.logger.info("[Intelligo-Face] Recognized: {} prob: {:.2f}".format(name, proba * 100))
                    # person = le.inverse_transform(j)
                    results[name] = proba
                elif len(self.new_embeddings) > 0:
                    # lookup in new embedding vector
                    distance = []
                    for em in self.new_embeddings:
                        distance.append(findCosineDistance(embedding, em))
                    min_idx = np.argmin(np.array(distance))

                    if distance[min_idx] < cosine_threshold:
                        self.logger.info(
                            "[Intelligo-Face] Recognized: {} from TEMPORAL LIST".format(self.new_knownnames[min_idx]))
                        results[self.new_knownnames[min_idx]] = 1 - distance[min_idx]
        # client upload a crop image with one face
        else:
            resized_image = cv2.resize(img, image_size, interpolation=cv2.INTER_AREA)
            embedding = self.rec_model.get_embedding(resized_image)
            embedding = np.reshape(embedding, (1, -1))
            # Predict class
            preds = self.classifier_model.predict(embedding)
            preds = preds.flatten()
            # Get the highest accuracy embedded vector
            j = np.argmax(preds)
            proba = preds[j]
            # Compare this vector to source class vectors to verify it is actual belong to this class
            match_class_idx = (self.labels == j)
            match_class_idx = np.where(match_class_idx)[0]
            selected_idx = np.random.choice(match_class_idx, comparing_num)
            compare_embeddings = self.embeddings[selected_idx]
            # Calculate cosine similarity
            cos_similarity = CosineSimilarity(embedding, compare_embeddings)
            self.logger.info("[Intelligo-Face] dist {} -- prob {}".format(cos_similarity, proba))
            if cos_similarity < cosine_threshold and proba > proba_threshold:
                name = self.le.classes_[j]
                # text = "{}".format(name)
                self.logger.info("[Intelligo-Face] Recognized: {} prob: {:.2f}".format(name, proba * 100))
                # person = le.inverse_transform(j)
                results[name] = proba
            elif len(self.new_embeddings) > 0:
                # lookup in new embedding vector
                distance = []
                for em in self.new_embeddings:
                    distance.append(findCosineDistance(embedding, em))
                min_idx = np.argmin(np.array(distance))

                if distance[min_idx] < cosine_threshold:
                    self.logger.info(
                        "[Intelligo-Face] Recognized: {} from TEMPORAL LIST".format(self.new_knownnames[min_idx]))
                    results[self.new_knownnames[min_idx]] = 1 - distance[min_idx]

        return results;