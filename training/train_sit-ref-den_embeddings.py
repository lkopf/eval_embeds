from gensim.models import word2vec
import pandas as pd
from itertools import combinations
from itertools import permutations
import gzip
import pickle
import json
from time import strftime
from collections import defaultdict
import os


# train situational embeddings for head nouns
# the context of a word are the nouns referring to other
# objects in the same scene
def situational_head(outfilename="./data/embeddings/sit_300dim.txt"):
    if os.path.isfile(outfilename):
        print("Outfile (%s) exists. Better check before I overwrite!"\
            % (outfilename))
        exit()

    with open("./training/Preproc/PreProcOut/heads_vecs_regions.pklz", "rb") as f:
        (heads2vec,region2head) = pickle.load(f)

    im2regions = defaultdict(list)
    for r in region2head:    
        imtup = tuple(r[:2])
        im2regions[imtup].append(r)

    # pair all heads that co-occur in an image
    im_cooc = []
    for im in im2regions.keys():
        rlist = im2regions[im]
        pcount = 0
        if len(rlist) > 1:
            for rperm in permutations(rlist):
                pcount += 1
                head_pairs = zip(*[region2head[rx] for rx in rperm])
                im_cooc += [list(tup) for tup in head_pairs]
                if pcount > 20:
                    break

    w2v_sit = word2vec.Word2Vec(im_cooc, vector_size=300, sg=2, window=3, min_count=5, workers=2, epochs=4) 
    w2v_sit.wv.save_word2vec_format(outfilename, binary=False)


def load_df():

    with gzip.open("./training/Preproc/PreProcOut/refcoco_refdf.pklz", "r") as f:     
        rrefdf = pickle.load(f,encoding="latin1")

    with gzip.open("./training/Preproc/PreProcOut/refcocoplus_refdf.pklz", "r") as f:
        r2refdf = pickle.load(f, encoding="latin1")
    with open("./training/Preproc/PreProcOut/refcoco_splits.json", "r") as f:
        rsplits = json.load(f)

    with gzip.open("./training/Preproc/PreProcOut/saiapr_refdf.pklz", "r") as f:
        srefdf = pickle.load(f, encoding="latin1")
    with open("./training/Preproc/PreProcOut/saiapr_90-10_splits.json", "r") as f:
        ssplits = json.load(f)

    with gzip.open("./training/Preproc/PreProcOut/grex_refdf.pklz", "r") as f:
        grefdf = pickle.load(f, encoding="latin1")
    with open("./training/Preproc/PreProcOut/google_refexp_rexsplits.json", "r") as f:
        gsplits = json.load(f)

    fulldf = pd.concat([grefdf,rrefdf,srefdf,r2refdf])

    return fulldf


# train standard w2v model on referring expressions
# predict left and right context of a word
def textual(outfilename="./data/embeddings/ref_300dim.txt"):
    if os.path.isfile(outfilename):
        print("Outfile (%s) exists. Better check before I overwrite!"\
            % (outfilename))
        exit()  

    fulldf = load_df()
    allrefs = [ref.split() for ref in fulldf.refexp]
    w2v_ref = word2vec.Word2Vec(allrefs, vector_size=300, sg=2, window=5, min_count=5, workers=2, epochs=4)
    w2v_ref.wv.save_word2vec_format(outfilename, binary=False)


# train denotational embeddings
# the context of a word are other words used to refer to the same object/region
def denotational(outfilename="./data/embeddings/den_300dim.txt"):
    if os.path.isfile(outfilename):
        print("Outfile (%s) exists. Better check before I overwrite!"\
            % (outfilename))
        exit()

    # group the data frame by regions
    fulldf = load_df()
    rgb = fulldf.groupby(["i_corpus", "image_id","region_id"])

    # ... simply pair each word with every other word from the same region
    concat_ref = []
    for k in rgb.groups.keys():
        reflist = [ref for ref in rgb.get_group(k)["refexp"]]
        if len(reflist) > 1:
            for rcomb in combinations(reflist,2):
                concat_ref += [(w1,w2) for w1 in rcomb[0].split() for w2 in rcomb[1].split()]
    w2v_den_all = word2vec.Word2Vec(concat_ref, vector_size=300, sg=2, window=1, min_count=5, workers=2, epochs=4)
    w2v_den_all.wv.save_word2vec_format(outfilename, binary=False)

   
if __name__ == "__main__":

    print("Start training embeddings...", strftime("%Y-%m-%d %H:%M:%S"))
    
    situational_head()
    print("Save situational embeddings.", strftime("%Y-%m-%d %H:%M:%S"))

    textual()
    print("Save textual embeddings.", strftime("%Y-%m-%d %H:%M:%S"))

    denotational()
    print("Save denotational embeddings.", strftime("%Y-%m-%d %H:%M:%S"))

