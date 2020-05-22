from pymongo import MongoClient
from bson.binary import Binary
import time
import pickle

embedding_file = './models/embeddings.pickle'


# receive two list of names and embedding vectors
def insert(names, embeddings, key):
    if len(names) != len(embeddings):
        return False
    else:
        # default localhost:27017 -- database=face_recognition
        db = MongoClient().face_recognition
        now = time.time()
        for n, e in zip(names, embeddings):
            doc = {'name': n,
                   'embedding': Binary(pickle.dumps(e, protocol=2)),
                   'timestamp': now,
                   'apikey': key}
            db.features.insert_one(dict(doc))
        return True


def save2file(key):
    # default localhost:27017 -- database=face_recognition
    db = MongoClient().face_recognition
    feature_vectors = []
    known_names = []

    # get all feature vectors and names
    results = db.features.find({'apikey': key})
    for doc in results:
        known_names.append(doc['name'])
        feature_vectors.append(pickle.loads(doc['embedding']))

    data = {"embeddings": feature_vectors, "names": known_names}
    # save to files
    with open(embedding_file, 'wb') as f:
        f.write(pickle.dumps(data))
