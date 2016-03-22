import os
import sys

import scipy as sp

from sklearn.feature_extraction.text import CountVectorizer

from utils import DATA_DIR

TOY_DIR = os.path.join(DATA_DIR, "toy")
posts = [open(os.path.join(TOY_DIR, f)).read() for f in os.listdir(TOY_DIR)]

new_post = "imaging databases"

vectorizer=CountVectorizer(min_df=1)
X_train = vectorizer.fit_transform(posts)

num_samples, num_features = X_train.shape
print("#samples: %d, #features: %d" % (num_samples, num_features))

new_post_vec = vectorizer.transform([new_post])
print(new_post_vec, type(new_post_vec))
print(new_post_vec.toarray())
print(vectorizer.get_feature_names())

def dist_raw(v1, v2):
    delta = v1 - v2
    return sp.linalg.norm(delta.toarray())

def dist_norm(v1,v2):
    v1_normalized = v1 / sp.linalg.norm(v1.toarray())
    v2_normalized = v2 / sp.linalg.norm(v2.toarray())

    delta = v1_normalized - v2_normalized

    return sp.linalg.norm(delta.toarra())

best_doc = None
best_dist = sys.maxsize
best_i = None

for i in range(0,num_samples):
    post = posts[i]
    if post == new_post:
        continue

    post_vec = X_train.getrow(i)
    d = dist(post_vec, new_post_vec)
