import pandas as pd
import numpy as np
import gzip
import pickle
import json

from time import localtime, strftime
from scipy.spatial.distance import pdist, squareform
from itertools import combinations
from collections import Counter

import os
from time import strftime
from gensim.models import word2vec
from gensim.models import KeyedVectors

import scipy.stats
import scipy
from copy import deepcopy
from scipy.spatial.distance import cdist
from itertools import combinations


def load_eval_data(wordlist):

    mensim = pd.read_csv("./data/datasets/MEN_dataset_lemma_form_full",sep=" ",header=None)
    print("All MEN", len(mensim))
    men_tuples = [(row[0][:-2],row[1][:-2],row[2]) for e,row in mensim.iterrows() \
     if (row[0][:-2] in wordlist and row[1][:-2] in wordlist)]

    print("MEN pairs", len(men_tuples))

    with open("./data/datasets/similarity_judgements.txt", "r") as f:
        silb0 = f.readlines()
        silb1 = [l.split() for l in silb0]
        silb2 = [(a[0],a[1],a[2]) for a in silb1 if len(a) == 3]
        print("All silberer", len(silb2))
        silb3 = [(a.split("#")[0],a.split("#")[1],b,c) for a,b,c in silb2[1:]]
        silb = [(a.split("_")[0],b,c,d) for (a,b,c,d) in silb3]
        #silb
        silb_sem = [(a,b,c) for (a,b,c,d) in silb if (a in wordlist and b in wordlist)]

        silb_vis = [(a,b,d) for (a,b,c,d) in silb if (a in wordlist and b in wordlist)]

        print("Silberer pairs", len(silb_sem))

    compatsim = pd.read_csv("./data/datasets/compatibility_ds.csv",sep=",")

    print("All compatibility", len(compatsim))
    compattuples = [(row["word1"],row["word2"],row["compatibility_mean"]) for e,row in compatsim.iterrows() \
     if (row["word1"] in wordlist and row["word2"] in wordlist)]


    print("Compatibility pairs", len(compattuples))

    return men_tuples, silb_sem, silb_vis, compattuples

def correlate_w2v(model,eval_set):
    y_pred = [model.similarity(w1,w2) for (w1,w2,j) in eval_set]
    y_true = [j for (w1,w2,j) in eval_set]
    corr,_ = scipy.stats.spearmanr(y_true,y_pred)
    return corr


def correlate_mat(simmat,word2ind,eval_set):
    y_pred = [simmat[word2ind[w1]][word2ind[w2]] for (w1,w2,j) in eval_set]
    y_true = [j for (w1,w2,j) in eval_set]
    corr,_ = scipy.stats.spearmanr(y_true,y_pred)
    return corr

def correlate_vecdict(model,eval_set):
    y_pred = [ 1-cdist([model[w1]],[model[w2]], "cosine")[0] for (w1,w2,j) in eval_set]
    y_true = [j for (w1,w2,j) in eval_set]
    corr,_ = scipy.stats.spearmanr(y_true,y_pred)
    return corr

def correlate_2_w2v(model1,model2,eval_set):
    y1 = [model1.similarity(w1,w2) for (w1,w2) in eval_set]
    y2 = [model2.similarity(w1,w2) for (w1,w2) in eval_set]
    corr,_ = scipy.stats.spearmanr(y1,y2)
    return corr

def correlate_w2v_dict(model1,model2,eval_set):
    y1 = [model1.similarity(w1,w2) for (w1,w2) in eval_set]
    y2 = [ 1-cdist([model2[w1]],[model2[w2]], "cosine")[0] for (w1,w2) in eval_set]
    corr,_ = scipy.stats.spearmanr(y1,y2)
    return corr

def eval_sim_judgements():
    w2v_ref = KeyedVectors.load_word2vec_format("./data/embeddings/ref_300dim.txt", binary=False)
    w2v_sit = KeyedVectors.load_word2vec_format("./data/embeddings/sit_300dim.txt", binary=False)
    w2v_den = KeyedVectors.load_word2vec_format("./data/embeddings/den_300dim.txt", binary=False)
    vis_av = KeyedVectors.load_word2vec_format("./data/embeddings/vis_1031dim.txt", binary=False)

    w2vden_visav = {}
    for word in w2v_den.index_to_key:
        if (word in vis_av):
            w2vden_visav[word] = np.hstack([w2v_den[word],vis_av[word]])
       

    vocab = set(w2v_den.index_to_key) & set(vis_av.index_to_key) & set(w2v_sit.index_to_key) & set(w2v_ref.index_to_key)
    print("Vocab", len(vocab))

    baroni = {}
    for line in open("./data/embeddings/baroni_400dim.txt"):
        l = line.split()
        w = l[0]
        if w in vocab: 
            baroni[w] = np.array(l[1:]).astype(float)   


    model_names = ["w2v_ref","w2v_den","w2v_sit","vis_av","baroni"]
    models = [w2v_ref,w2v_den,w2v_sit,vis_av,baroni]

    eval_sets = load_eval_data(vocab)

    results = []
    for x,model in enumerate(models):
        if x < 3:
            corrs = [correlate_w2v(model,ev) for ev in eval_sets]
        elif x < 5:
            corrs = [correlate_vecdict(model,ev) for ev in eval_sets]
        results.append([model_names[x]]+["%.3f"%c for c in corrs])

    df = pd.DataFrame(results,columns=["Model","MEN","SemSim","VisSim","Compatibility"])
    print(df.to_latex(index=False))

    print("MODEL FUSION VIA CONCATENATION")

    w2v_densit = {}
    w2v_denref = {}
    w2v_densitref = {}
    w2v_sitref = {}

    for word in w2v_den.index_to_key:
        if (word in w2v_sit) and (word in w2v_ref):
            w2v_denref[word] = np.hstack([w2v_den[word],w2v_ref[word]])
            w2v_densit[word] = np.hstack([w2v_sit[word],w2v_den[word]])
            w2v_sitref[word] = np.hstack([w2v_sit[word],w2v_ref[word]])
            w2v_densitref[word] = np.hstack([w2v_den[word],w2v_sit[word],w2v_ref[word]])

    model_names = ["w2v_den|ref","w2v_den|sit","w2v_ref|sit","w2v_den|sit|ref"]

    results = []
    for x,model in enumerate([w2v_denref,w2v_densit,w2v_sitref,w2v_densitref]):
        corrs = [correlate_vecdict(model,ev) for ev in eval_sets]
        results.append([model_names[x]]+["%.3f"%c for c in corrs])

    df = pd.DataFrame(results,columns=["Model","MEN","SemSim","VisSim","Compatibility"])
    print(df.to_latex(index=False))

    return df

def get_hyper_direction_acc(wordmodel,hyper):

    acc = 0
    for w1,w2 in hyper:
        if w1 in wordmodel and w2 in wordmodel:
            e1 = scipy.stats.entropy(np.exp(wordmodel[w1]))
            e2 = scipy.stats.entropy(np.exp(wordmodel[w2]))
            if (1-(e1/e2)) > 0:
                acc += 1
        else:
            print(w1,w2)
    
    return acc/len(hyper)


def eval_hyper():
    w2v_ref = KeyedVectors.load_word2vec_format("./data/embeddings/ref_300dim.txt", binary=False)
    w2v_sit = KeyedVectors.load_word2vec_format("./data/embeddings/sit_300dim.txt", binary=False)
    w2v_den = KeyedVectors.load_word2vec_format("./data/embeddings/den_300dim.txt", binary=False)
    vis_av = KeyedVectors.load_word2vec_format("./data/embeddings/vis_1031dim.txt", binary=False)

    w2v_densit = {}
    w2v_denref = {}
    w2v_densitref = {}

    for word in w2v_den.index_to_key:
        if (word in w2v_sit) and (word in w2v_ref):
            w2v_denref[word] = np.hstack([w2v_den[word],w2v_ref[word]])
            w2v_densit[word] = np.hstack([w2v_sit[word],w2v_den[word]])
            w2v_densitref[word] = np.hstack([w2v_den[word],w2v_sit[word],w2v_ref[word]])


    w2vden_visav = {}
    for word in w2v_den.index_to_key:
        if (word in vis_av):
            w2vden_visav[word] = np.hstack([w2v_den[word],vis_av[word]])
       

    vocab = set(w2v_densitref.keys()) & set(vis_av.index_to_key)
    print("Vocab", len(vocab))

    hyper = []
    for line in open("./data/datasets/BLESS.txt"):
        if "\thyper\t" in line:
            l = line.split()
            w1 = l[0][:-2]
            w2 = l[3][:-2]
            if w1 in vocab and w2 in vocab:
                hyper.append((w1,w2))


    baroni = {}
    for line in open("./data/embeddings/baroni_400dim.txt"):
        l = line.split()
        w = l[0]
        if w in w2v_ref.index_to_key and l[1] != "more": 
            baroni[w] = np.array(l[1:]).astype(float)

    print("N hyper pairs", len(hyper))
    model_names = ["w2v_ref","w2v_den","w2v_sit","w2v_den|ref","w2v_den|sit",\
    "w2v_den|sit|ref","vis_av","w2vden_visav","baroni"]

    results = []
    for x,model in enumerate([w2v_ref,w2v_den,w2v_sit,w2v_denref,w2v_densit,\
        w2v_densitref,vis_av,w2vden_visav,baroni]):
        acc = get_hyper_direction_acc(model,hyper)
        results.append([model_names[x], "%.2f"%(acc*100)])

    df = pd.DataFrame(results,columns=["Model","Hyper. Direction"])
    print(df.to_latex(index=False))
    return df


def model_correlations():
    w2v_ref = KeyedVectors.load_word2vec_format("./data/embeddings/ref_300dim.txt", binary=False)
    w2v_sit = KeyedVectors.load_word2vec_format("./data/embeddings/sit_300dim.txt", binary=False)
    w2v_den = KeyedVectors.load_word2vec_format("./data/embeddings/den_300dim.txt", binary=False)
    vis_av = KeyedVectors.load_word2vec_format("./data/embeddings/vis_1031dim.txt", binary=False)

    vocab = set(w2v_den.index_to_key) & set(vis_av.index_to_key) & \
    set(w2v_ref.index_to_key) & set(w2v_sit.index_to_key)
    print("Vocab", len(vocab))
    eval_sets = load_eval_data(vocab)
    pairs = []
    for ev in eval_sets:
        pairs += [(w1,w2) for (w1,w2,_) in ev]

    print("Pairs", len(pairs))

    c1 = correlate_2_w2v(w2v_ref,w2v_den,pairs)
    c2 = correlate_2_w2v(w2v_ref,w2v_sit,pairs)
    c3 = correlate_w2v_dict(w2v_ref,vis_av,pairs)
    c4 = correlate_w2v_dict(w2v_den,vis_av,pairs)
    c5 = correlate_2_w2v(w2v_den,w2v_sit,pairs)
    c6 = correlate_w2v_dict(w2v_sit,vis_av,pairs)

    results = []
    results.append(("txt",1,"%.2f"%c1,"%.2f"%c2,"%.2f"%c3))
    results.append(("den","%.2f"%c1,1,"%.2f"%c5,"%.2f"%c4))
    results.append(("sit","%.2f"%c2,"%.2f"%c4,1,"%.2f"%c6))
    results.append(("vis","%.2f"%c3,"%.2f"%c4,"%.2f"%c6,1))

    df = pd.DataFrame(results,columns=["Model","txt","den","sit","vis"])
    print(df.to_latex(index=False))
    return df

print(strftime("%Y-%m-%d %H:%M:%S", localtime()))
print("Start evaluating embeddings...\n")

print("CORRELATION between models")
model_correlations()

print("CORRELATION WITH SIMILARITY JUDGEMENTS")
eval_sim_judgements()

print("PREDICTION OF HYPERNYMY DIRECTION")
eval_hyper()

print(strftime("%Y-%m-%d %H:%M:%S", localtime()))
print("Done!")

