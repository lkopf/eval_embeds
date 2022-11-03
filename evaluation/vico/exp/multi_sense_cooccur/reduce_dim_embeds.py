import os
import numpy as np
import h5py
import umap

import evaluation.vico.utils.io as io


def main(exp_const,data_const):
    print('Loading embeddings ...')
    embed = io.load_h5py_object(
        data_const.word_vecs_h5py)['embeddings'][()]

    X_embed = umap.UMAP(n_components=100).fit_transform(embed)

    embeddings = np.stack(X_embed)
    print('Embedding matrix shape:',embeddings.shape)
    mean = np.mean(embeddings,axis=0)
    std = np.std(embeddings,axis=0)
    min_ = np.min(embeddings,axis=0)
    max_ = np.max(embeddings,axis=0)

    print(f'Creating {exp_const.embed_name}_100dim.h5py ...')
    w2v_h5py = h5py.File(
        os.path.join(
            exp_const.exp_dir,
            f'{exp_const.embed_name}_100dim.h5py'),
        'w')                 

    w2v_h5py.create_dataset('embeddings',data=embeddings)
    w2v_h5py.create_dataset('mean',data=mean)
    w2v_h5py.create_dataset('std',data=std)
    w2v_h5py.create_dataset('min',data=min_)
    w2v_h5py.create_dataset('max',data=max_)
    w2v_h5py.close()
