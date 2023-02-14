from __future__ import division

import json
import os

import numpy as np
import pandas as pd

import cPickle as pickle
import gzip

import re
from nltk import pos_tag
from collections import Counter,defaultdict

from nltk.tag import CRFTagger
tagger = ct = CRFTagger()
ct.set_model_file("crfpostagger") # Julian Hough's tagger trained on switchboard
                                  # might be useful for tagging NPs, but we don't know ...


# some words which are often tagged as nouns, but we know that they are not heads/object names
NOHEAD = ['left','right','middle','center','corner','front','rightmost','leftmost','back',\
         'blue','red','white','purple','green','gray','grey','brown','blond','black','yellow',\
         'tall','standing','empty','tan','bald','silver','leftest','far','farthest','furthest',\
         'whole','lol','sorry','tallest','nearest','round','taller','righter','rightest','background',\
          'top','okay','pic','part','bottom','click','mid','closest','teal','hot','wooden',\
         'short','shortest','bottm','topmost','dirty','spiders','wtf','skinny','fat','sexy','jumping',
         'frontmost','yes','wearing','lil','creepy','serving','beside','beige','upper','lower','side',
         'facing','blurry','color','colour','dark','pink','foreground','facing','holding']


NOUNS = open('noun_list_long.txt').readlines()

RELUNI = ['below',
            'above',
            'between',
            'not',
            'behind',
            'under',
            'underneath',
            'by',
            'near',
            'with',
            'at',
            'that',
            'who',
            'beside',
            'besides']

RELOF = ['front', 'right', 'left', 'ontop', 'top', 'middle','side','out']

RELTO = ['next', 'close', 'closest']




with gzip.open('../Preproc/PreProcOut/refcoco_refdf.pklz', 'r') as f:
    rrefdf = pickle.load(f)

with gzip.open('../Preproc/PreProcOut/saiapr_refdf.pklz', 'r') as f:
    srefdf = pickle.load(f)

with gzip.open('../Preproc/PreProcOut/grex_refdf.pklz', 'r') as f:
    grefdf = pickle.load(f)

with gzip.open('../Preproc/PreProcOut/refcocoplus_refdf.pklz', 'r') as f:
    r2refdf = pd.read_pickle(f)
    r2refdf = pd.DataFrame(r2refdf)

fulldf = pd.concat([grefdf,rrefdf,srefdf,r2refdf])

WORDS = Counter(' '.join(list(fulldf['refexp'])).split())

# pattern-pos-based hack
# that identifies the head of a refex
def get_head_noun(tagged):
    
    for (word,pos) in tagged:
        if len(word) > 1 and \
         not word in NOHEAD and \
         WORDS[word] > 1:
            if pos == 'NN':
                return word
            if pos == 'NNS':
                return word
            if word in NOUNS:
                return word
    #print tagged
    return ""

# getting the index of the relational prep.
# if refex is relational at all
def get_rel_index(tagged):
    for ix,(word,_) in enumerate(tagged):
        if word in RELUNI:
            return ix
        if word in RELOF:
            if ix+1 < len(tagged):
                if tagged[ix+1][0] == 'of':
                    return ix+1

        if word in RELTO:
            if ix+1 < len(tagged):
                if tagged[ix+1][0] == 'to':
                    return ix+1

    return -1

def crf_tagged(refexp):
    ref = []
    for w in refexp.split():
        try:
            ref.append(unicode(w))
        except:
            pass
    return ct.tag(ref)

fulldf['relindex'] = fulldf['tagged'].apply(lambda x: get_rel_index(x))
fulldf['crf_tagged'] = fulldf['refexp'].apply(lambda x: crf_tagged(x))

# looking for heads
headlist = []
HEADS = Counter()
for x,row in fulldf.iterrows():

    # if taggers disagree, we simply collect both heads
    if row['relindex'] > -1:
        refex1 = row['tagged'][:row['relindex']]
        refex2 = row['crf_tagged'][:row['relindex']]
    else:
        refex1 = row['tagged']
        refex2 = row['crf_tagged']
    #print x,row['relindex'],
    #print refex
    h1 = get_head_noun(refex1)
    h2 = get_head_noun(refex2)
    
    headlist.append([h1,h2])
    if len(h1) > 0:
        HEADS[h1] += 1
    if len(h2) > 0:
        HEADS[h2] += 1


fulldf['heads'] = headlist

with open('../Preproc/PreProcOut/fulldf_with_heads.pklz', 'w') as f:
    pickle.dump(fulldf,f)

# get w2v vector for head words with freq > 10
HEADS2VEC = {}
for line in open('../Data/EN-wform.w.5.cbow.neg10.400.subsmpl.txt'):
    l = line.split()
    w = l[0]
    if HEADS[w] > 10:
        HEADS2VEC[w] = np.array(l[1:]).astype(float)


# get a list of heads for each object in our data
# ... these heads have freq > 10
# ... occur in the w2v vocabulary
region2head = defaultdict(list)
alltuples = fulldf[['i_corpus','image_id','region_id','heads']].values
for tup in alltuples:
    for h in tup[3]:
        if h in HEADS2VEC:
            region2head[tuple(tup[:3])].append(h)

print "These heads have freq > 10, but will be ignored because they don't have a w2v vector:"
print [h for h in HEADS if HEADS[h] > 10 and h not in HEADS2VEC]

print "Final number of heads", len(HEADS2VEC)
print "Final number of regions with a head", len(region2head)

with open('../Preproc/PreProcOut/heads_vecs_regions.pklz', 'w') as f:
    pickle.dump((HEADS2VEC,region2head),f)

