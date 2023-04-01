import os
import csv
import numpy as np
import pandas as pd
from gensim import utils
from gensim.models import KeyedVectors

# Load embeddings
w2v_ref = KeyedVectors.load_word2vec_format("./data/embeddings/ref_300dim.txt", binary=False)
w2v_sit = KeyedVectors.load_word2vec_format("./data/embeddings/sit_300dim.txt", binary=False)
w2v_den = KeyedVectors.load_word2vec_format("./data/embeddings/den_300dim.txt", binary=False)
w2v_vis = KeyedVectors.load_word2vec_format("./data/embeddings/vis_1031dim.txt", binary=False)
w2v_denref = KeyedVectors.load_word2vec_format("./data/embeddings/denref_600dim.txt", binary=False)
w2v_baroni = KeyedVectors.load_word2vec_format("./data/embeddings/baroni_400dim.txt", binary=False)

embeds = w2v_vis, w2v_ref, w2v_den, w2v_sit, w2v_denref, w2v_baroni

results = []
similarity_results = []
with open('./data/datasets/analogy_test.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        row_results = []
        row_similarities = []
        for embed in embeds:
            # a : b :: c :: ?
            a, b, c = row[0], row[1], row[2]
            # find prediction for d
            result = embed.most_similar_cosmul(positive=[c, b], negative=[a])
            most_similar_key, similarity = result[0]  # look at the first match
            row_results.append(most_similar_key)
            row_similarities.append(similarity)
        results.append(row_results)
        similarity_results.append(row_similarities)

df = pd.DataFrame(results,columns=["vis", "ref", "den", "sit", "denref", "baroni"])
print(df.to_latex(index=False))        

df_sim = pd.DataFrame(similarity_results,columns=["vis", "ref", "den", "sit", "denref", "baroni"])
print(df_sim.to_latex(index=False)) 

print(df_sim.mean())