import pickle
import numpy as np
import gzip
from gensim import utils
from gensim.models.keyedvectors import Word2VecKeyedVectors
import os
import h5py
import json 

from gensim.models import KeyedVectors

"""
Convert all embeddings to .txt and .h5py format. Collect word_to_idx in .json format.
These formats are needed for Vico evaluation methods
"""

with gzip.open("vis_av_refvocab.pklz", "r") as f:
    vis_av = pickle.load(f, encoding="latin1")

with open("w2v_denref.pkl", "rb") as f:
    w2v_denref = pickle.load(f, encoding="latin1")


def convert_word2vec_to_txt(fname, vocab, vectors, binary=True, total_vec=2):
    """Store the input-hidden weight matrix in the same format used by the original
    C word2vec-tool, for compatibility.

    Parameters
    ----------
    fname : str
        The file path used to save the vectors in.
    vocab : dict
        The vocabulary of words.
    vectors : numpy.array
        The vectors to be stored.
    binary : bool, optional
        If True, the data wil be saved in binary word2vec format, else it will be saved in plain text.
    total_vec : int, optional
        Explicitly specify total number of vectors
        (in case word vectors are appended with document vectors afterwards).

    """
    if not (vocab or vectors):
        raise RuntimeError("no input")
    if total_vec is None:
        total_vec = len(vocab)
    vector_size = vectors.shape[1]
    assert (len(vocab), vector_size) == vectors.shape
    with utils.open(fname, "wb") as fout:
        fout.write(utils.to_utf8("%s %s\n" % (total_vec, vector_size)))
        # store in sorted order: most frequent words at the top
        for word, row in vocab.items():
            if binary:
                row = row.astype("float32")
                fout.write(utils.to_utf8(word) + b" " + row.tostring())
            else:
                fout.write(utils.to_utf8("%s %s\n" % (word, " ".join(repr(val) for val in row))))

print("Convert visual embeddings")
vis = Word2VecKeyedVectors(vector_size=1031)
vis.key_to_index = vis_av
vis.vectors = np.array(list(vis_av.values()))
convert_word2vec_to_txt(binary=False, fname="./data/embeddings/vis_1031dim.txt", total_vec=len(vis_av), vocab=vis.key_to_index, vectors=vis.vectors)

print("Convert denref embeddings")
denref = Word2VecKeyedVectors(vector_size=600)
denref.key_to_index = w2v_denref
denref.vectors = np.array(list(w2v_denref.values()))
convert_word2vec_to_txt(binary=False, fname="./data/embeddings/denref_600dim.txt", total_vec=len(w2v_denref), vocab=denref.key_to_index, vectors=denref.vectors)

print("Deleting pickle files...")
os.remove("vis_av_refvocab.pklz")
os.remove("w2v_denref.pkl")


# get embeddings in h5py and json format
def write(file_name, data, mode="wb"):
    with open(file_name, mode) as f:
        f.write(data)

def dump_json_object(dump_object, file_name, compress=False, indent=4):
    data = json.dumps(
        dump_object, cls=NumpyAwareJSONEncoder, sort_keys=True, indent=indent)
    if compress:
        write(file_name, gzip.compress(data.encode("utf8")))
    else:
        write(file_name, data, "w")

class NumpyAwareJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            if obj.ndim == 1:
                return obj.tolist()
            else:
                return [self.default(obj[i]) for i in range(obj.shape[0])]
        elif isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, np.int32):
            return int(obj)
        elif isinstance(obj, np.int16):
            return int(obj)
        elif isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.float16):
            return float(obj)
        elif isinstance(obj, np.uint64):
            return int(obj)
        elif isinstance(obj, np.uint32):
            return int(obj)
        elif isinstance(obj, np.uint16):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


embed_names = ["baroni_400dim", "den_300dim", "denref_600dim", "ref_300dim", "sit_300dim", "vis_1031dim"]
embed_path = "./data/embeddings/"

for embed in embed_names:
    if embed == "baroni_400dim":
        w2v_ref = KeyedVectors.load_word2vec_format(f"{embed_path}ref_300dim.txt", binary=False)
        f = open(f"{embed_path}{embed}.txt","r")
        embeddings = []
        word_to_idx = {}
        next(f)
        _count = 0
        for i, line in enumerate(f):
            splitLine = line.split()
            word = splitLine[0]
            if word in w2v_ref.index_to_key and splitLine[1] != "more":        
                embeddings.append(np.array([float(val) for val in splitLine[1:]]))
                word_to_idx[word] = _count
                _count += 1
    else: # den, denref, ref, sit, vis
        f = open(f"{embed_path}{embed}.txt","r")
        embeddings = []
        word_to_idx = {}
        next(f)
        for i,line in enumerate(f):
            splitLine = line.split()
            word = splitLine[0]
            embeddings.append(np.array([float(val) for val in splitLine[1:]]))
            word_to_idx[word] = i
    
    embeddings = np.stack(embeddings)
    print("Embedding matrix shape:", embeddings.shape)
    mean = np.mean(embeddings,axis=0)
    std = np.std(embeddings,axis=0)
    min_ = np.min(embeddings,axis=0)
    max_ = np.max(embeddings,axis=0)

    print(f"Creating {embed}.h5py ...")
    w2v_h5py = h5py.File(
        os.path.join(
            embed_path,
            f"{embed}.h5py"),
        "w")
    w2v_h5py.create_dataset("embeddings",data=embeddings)
    w2v_h5py.create_dataset("mean",data=mean)
    w2v_h5py.create_dataset("std",data=std)
    w2v_h5py.create_dataset("min",data=min_)
    w2v_h5py.create_dataset("max",data=max_)
    w2v_h5py.close()

    print(f"Saving {embed}_word_to_idx.json ...")
    dump_json_object(
        word_to_idx,
        os.path.join(embed_path,f"{embed}_word_to_idx.json"))

    print(f"Vocab size: {len(word_to_idx)}")
