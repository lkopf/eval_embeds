# coding: utf-8

from __future__ import division
from collections import Counter
from time import strftime

import numpy as np
import sklearn
from joblib import Parallel, delayed

# number of ID features at the beginning of X
#  so that [POSTIDINDEX:] indexes the true features 
POSTIDINDEX = 3

RELWORDS = ['below',
            'above',
            'between',
            'not',
            'behind',
            'under',
            'underneath',
            'front of',
            'right of',
            'left of',
            'ontop of',
            'next to',
            'middle of']

STOPWORDS = ['the', 'a', 'an']


def is_relational(expr):
    for rel in RELWORDS:
        if rel in expr:
            return True
    return False

def filter_relational_expr(refdf):
    '''View on given refdf with only non-relation refexps.'''
    return refdf[~(refdf['refexp'].apply(is_relational))]


def wordlist_min_freq(refdf, minfreq, stopped=True):
    '''Wordlist out of refdf; minimum frequency criterion.'''
    rexc = Counter(' '.join(refdf['refexp'].tolist()).split())
    list_ = [w for w,c in rexc.items() if c >= minfreq]
    if stopped:
        list_ = [w for w in list_ if w not in STOPWORDS]
    return list_

def wordlist_n_top(refdf, ntop, stopped=True):
    '''Wordlist out of refdf; n-most frequent criterion.'''
    rexc = Counter(' '.join(refdf['refexp'].tolist()).split())
    if stopped:
        list_ = rexc.most_common(ntop + len(STOPWORDS))
        list_ = [w for w,_ in list_ if w not in STOPWORDS][:ntop]
    else:
        list_ = [w for w,_ in rexc.most_common(ntop)]
    return list_

def wordlist_by_criterion(refdf, criterion, parameter, stopped=True):
    if criterion == 'min':
        return wordlist_min_freq(refdf, parameter, stopped=stopped)
    if criterion == 'ntop':
        return wordlist_n_top(refdf, parameter, stopped=stopped)



def create_word2den(refdf, wordlist, tagged=False):
    '''Given refdf and wordlist, returns dict of occurences (id triples) 
    of words.'''
    word2den = {}
    for _, row in refdf.iterrows():
        exprlist = row['refexp'].split()
        if tagged:
            exprlist = row['tagged']
        for word in exprlist:
            if word in wordlist:
                word_den_list = word2den.get(word, [])
                word2den[word] = word_den_list + [(row['i_corpus'],
                                                   row['image_id'],
                                                   row['region_id'])]
    
    return {k:list(set(v)) for k,v in word2den.items()}

    
def make_train_same(X, X_indices, wrd2dn, word, nneg):
    X_full_index, X_img_index = X_indices
    X_train = []
    y_train = []
    
    for this_full_id in wrd2dn[word]:
        this_corp_id, this_image_id, this_region_id = this_full_id

        # get the pos example
        if this_full_id not in X_full_index:
            print('no features for (%d %d %d)! skipped.' % (this_full_id))
            continue
        pos_index = X_full_index[(this_full_id)]
        pos_feats = [X[pos_index,3:]]

        X_train.append(pos_feats)
        y_train.append(True)

        # get negative examples

        # - take negative examples from same image:
        xfrom, xto = X_img_index[(this_corp_id,this_image_id)]
        neg_feats = X[xfrom:xto+1,3:]
        neg_feats = neg_feats[neg_feats[:,2] != this_region_id]

        if neg_feats.shape[0] == 0:
            print('  No neg samples from same image available.')
            print('  You should see this only rarely, otherwise better use method \'random\'')
            continue
        randix = np.random.choice(len(neg_feats), nneg)

        X_train.append(neg_feats[randix])
        y_train.extend([False] * len(randix))
    return X_train, y_train

def make_train_rand(X, X_indices, wrd2dn, word, nneg, nsrc):
    FIXEDRATIO = 2 # in mode "randmax", make ratio pos/neg
    # minimally this, if there are more than
    # 1/FIXEDRATIO number of positive instances. Ensures
    # that there is minimally this distance between pos and neg.
    
    X_full_index, X_img_index = X_indices
    X_train = []
    y_train = []

    # first create mask that identifies all positive
    #  instances
    mask = np.zeros(len(X), dtype=bool)
    for this_full_id in wrd2dn[word]:
        if this_full_id in X_full_index:
            mask[X_full_index[this_full_id]] = True
    this_X_train = X[mask,3:]
    X_train.append(this_X_train)
    y_train.extend([True] * len(this_X_train))

    # negative examples are all others (inverted mask)
    X_neg = X[~mask,3:]

    if nsrc == 'random':
        # nneg negative samples for each positive:
        n_neg_samples_total = nneg * len(this_X_train)
        X_train.append(X_neg[np.random.choice(len(X_neg), n_neg_samples_total)])
        y_train.extend([False] * n_neg_samples_total)
    elif nsrc == 'randmax':
        # new mode (2017-03-17): randmax selects fixed number of
        #  negative instances (up to nneg). This brings in notion
        #  of frequency: Frequent words (for which there are many
        #  positive instances) will have a higher ratio pos/neg, as
        #  neg is fixed for all.
        if nneg < len(X_neg):
            if nneg < FIXEDRATIO * len(this_X_train):
                nneg = min(FIXEDRATIO * len(this_X_train),
                           len(X_neg))
            X_train.append(X_neg[np.random.choice(len(X_neg), nneg)])
            y_train.extend([False] * nneg)
        else:
            X_train.append(X_neg)
            y_train.extend([False] * len(X_neg))
    elif nsrc == 'allneg':
        X_train.append(X_neg)
        y_train.extend([False] * len(X_neg))
    return X_train, y_train
    
def make_train(X, X_indices, wrd2dn, word, nneg, nsrc):
    '''Construct training feature set for word.'''
    
    if nsrc == 'same':
        X_train, y_train = make_train_same(X, X_indices, wrd2dn,
                                           word, nneg)
            
    elif nsrc == 'random' or nsrc == 'allneg' or nsrc == 'randmax':
        X_train, y_train = make_train_rand(X, X_indices, wrd2dn,
                                           word, nneg, nsrc)
    else:
        print("unknown training mode!")

    #print len(X_train), map(len, X_train)
    X_train = np.concatenate(X_train, axis=0)
    #print 'final shape: ', X_train.shape
    return tuple(sklearn.utils.shuffle(X_train, y_train))
    
def create_X_lookup_indices(X):
    N = int(len(X))
    # for full region ID (corpus, image, region):
    X_full_index = dict(zip([tuple(e) \
                             for e in X[:,:3].astype(int).tolist()], range(N)))
    # range for image (under the assumption that all regions of this image are
    #  in consecutive rows in X)
    # - upper index
    X_img_index = dict(zip([tuple(e) \
                            for e in X[:,:2].astype(int).tolist()], range(N)))
    # - lower index. Found by going through reversed X 
    X_img_index_rev = dict(zip([tuple(e) \
                                for e in X[::-1,:2].astype(int).tolist()],
                               range(N)))
    # - combine
    X_img_range_index = {k:[(X_img_index_rev[k] - N+1)*-1, X_img_index[k]]\
                         for k in X_img_index.keys()}
    return (X_full_index, X_img_range_index)

def train_this_word(X, X_indices, wrd2dn,
                    classifier, classf_params,
                    word, nneg, nsrc, n, wordlist):
    print(strftime("%Y-%m-%d %H:%M:%S"))
    print('[%d/%d] training classifier for \'%s\'...'
          % (n+1, len(wordlist), word))

    print('... assembling training data... [%s]'
          % strftime("%Y-%m-%d %H:%M:%S"))
    Xt, yt = make_train(X, X_indices, wrd2dn, word, nneg, nsrc)
    npos = np.sum(yt)
    print(' %s: (%d pos instances, %d neg)' % (word, npos, len(Xt)-npos))

    print('... fitting model... [%s]'\
          % strftime("%Y-%m-%d %H:%M:%S"))
    this_classf = classifier(**classf_params)
    this_classf.fit(Xt,yt)
    print('... done.')
    return  {'npos': npos,
             'clsf': this_classf,
             'nneg': nneg,
             'nsrc': nsrc}


def train_model(refdf, X, wordlist, classifier_spec,
                nneg=5, nsrc='same'):
    '''Train the WAC models, for wordlist.'''
    classifier, classf_params = classifier_spec
    wrd2dn = create_word2den(refdf, wordlist)
    X_indices = create_X_lookup_indices(X)
    # 2017-03-17: since parallelisation seems to gain nothing,
    #   and this will be run with default n_jobs=1, could roll it
    #   back again. Just use normal list comprehension here:
    # clsf_list = Parallel(n_jobs=n_jobs)(delayed(train_this_word)\
    #                                     (*[X, X_indices, wrd2dn,
    #                                        classifier, classf_params,
    #                                        word, nneg, nsrc, n, wordlist])
    #                                     for n, word in enumerate(wordlist))
    clsf_list = [train_this_word(X, X_indices, wrd2dn, classifier,
                                 classf_params, word, nneg, nsrc, n, wordlist)\
                 for n, word in enumerate(wordlist)]
    return {word:e for word, e in zip(wordlist, clsf_list)}


# from similarity_2017.py

def get_all_instances_of(X, X_full_index, wrd2dn, word):
    return X[[e for e in [X_full_index.get(e, None)
                          for e in wrd2dn[word]]
              if e is not None]][:, POSTIDINDEX:]

# Visual averages
def get_mean_visual_feats(X, X_full_index, wrd2dn, wordlist):
    average_X = []
    for word in wordlist:
        average_X.append(np.mean(get_all_instances_of(X, X_full_index,
                                                      wrd2dn, word), axis=0))
    return np.array(average_X)