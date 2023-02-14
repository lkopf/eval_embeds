# coding: utf-8

from __future__ import division
import pandas as pd
import numpy as np
import gzip
import pickle
import json

from scipy.spatial.distance import pdist, squareform

from itertools import combinations

from collections import Counter

import os
from time import strftime

from gensim.models import KeyedVectors

import sys
sys.path.append("./training/TrainModels")
sys.path.append("./training/Utils")

from utils import filter_by_filelist, filter_X_by_filelist
from train_model import STOPWORDS, is_relational, create_word2den, create_X_lookup_indices
from train_model import filter_relational_expr, create_X_lookup_indices, make_train
from train_model import POSTIDINDEX, get_mean_visual_feats


with open("./training/Preproc/PreProcOut/refcoco_splits.json", "r") as f:
    splits = json.load(f)

with open("./training/Preproc/PreProcOut/saiapr_90-10_splits.json", "r") as f:
    ssplit90 = json.load(f)

# tlist_a = ssplit90["train"] ###
# tlist_b = splits["train"] ###

print("loading up refdf", strftime("%Y-%m-%d %H:%M:%S"))

with gzip.open("./training/Preproc/PreProcOut/saiapr_refdf.pklz", "r") as f:
    refdf_a = pickle.load(f)
# refdf_a = filter_by_filelist(refdf_a, tlist_a) ###

with gzip.open("./training/Preproc/PreProcOut/refcoco_refdf.pklz", "r") as f:
    refdf_b = pickle.load(f)
# refdf_b = filter_by_filelist(refdf_b, tlist_b) ###


with gzip.open("./training/Preproc/PreProcOut/refcocoplus_refdf.pklz", "r") as f:
    refdf_c = pickle.load(f)
# refdf_c = filter_by_filelist(refdf_c, tlist_b) ###

refdf_full = pd.concat([refdf_a, refdf_b, refdf_c])

w2v_ref = KeyedVectors.load_word2vec_format("./data/embeddings/ref_300dim.txt", binary=False)

wordlist = w2v_ref.index_to_key # if using KeyedVectors
wrd2dn = create_word2den(refdf_full, wordlist)
wordlist = [w for w in wordlist if w in wrd2dn]

print("size wordlist", len(wordlist))

print("loading up feature matrix", strftime("%Y-%m-%d %H:%M:%S"))

X_a = np.load("./training/ExtractFeats/ExtrFeatsOut/saiapr.npz")
X_a = X_a["arr_0"]

# X_a = filter_X_by_filelist(X_a, tlist_a) ###

X_b = np.load("./training/ExtractFeats/ExtrFeatsOut/mscoco.npz")
X_b = X_b["arr_0"]

# X_b = filter_X_by_filelist(X_b, tlist_b) ###


X_full = np.concatenate([X_a, X_b])

X_full_index, _X_img_index = create_X_lookup_indices(X_full)

print("loading up vis. av.", strftime("%Y-%m-%d %H:%M:%S"))


mean_vis_feats = get_mean_visual_feats(X_full, X_full_index,
                                               wrd2dn, wordlist)

avdict = {word:mean_vis_feats[wx] for wx,word in enumerate(wordlist)}
# with gzip.open("./training/vis_av_refvocab_traindf.pklz", "w") as f: ###
with gzip.open("./training/vis_av_refvocab.pklz", "w") as f:
    pickle.dump(avdict, f)

print("Save visual embeddings.", strftime("%Y-%m-%d %H:%M:%S"))

w2v_den = KeyedVectors.load_word2vec_format("./data/embeddings/den_300dim.txt", binary=False)

w2v_denref = {}

for word in w2v_den.index_to_key:
    if word in w2v_ref:
        w2v_denref[word] = np.hstack([w2v_den[word],w2v_ref[word]])

# save concatenated embeddings as pickle file
with open("./training/w2v_denref.pkl", "wb") as file:    
    pickle.dump(w2v_denref, file)

print("Save denref embeddings.", strftime("%Y-%m-%d %H:%M:%S"))
