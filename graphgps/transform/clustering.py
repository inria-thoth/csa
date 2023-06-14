# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import faiss
import numpy as np
import torch


def preprocess_features(npdata, pca=256):
    """Preprocess an array of features.
    Args:
        npdata (np.array N * ndim): features to preprocess
        pca (int): dim of output
    Returns:
        np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
    """
    _, ndim = npdata.shape
    npdata =  npdata.astype('float32')

    # Apply PCA-whitening with Faiss
    mat = faiss.PCAMatrix (ndim, pca, eigen_power=-0.5)
    mat.train(npdata)
    assert mat.is_trained
    npdata = mat.apply_py(npdata)

    # L2 normalization
    row_sums = np.linalg.norm(npdata, axis=1)
    npdata = npdata / row_sums[:, np.newaxis]

    return npdata


def run_kmeans(x, nmb_clusters, verbose=True):
    """Runs kmeans on 1 GPU.
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster
    """
    niter = 20
    d = x.shape[1]
    kmeans = faiss.Kmeans(d, nmb_clusters, niter=niter, verbose=verbose, max_points_per_centroid=1000000)
    kmeans.seed = np.random.randint(1234)
    kmeans.train(x)

    # This will return the nearest centroid for each line vector in x in I. D contains the squared L2 distances.
    D, I = kmeans.index.search(x, 1)
    # import pdb; pdb.set_trace()

    # stats = clus.iteration_stats
    # obj = np.array([stats.at(i).obj for i in range(stats.size())])
    # losses = faiss.vector_to_array(obj)
    # if verbose:
    #     print('k-means loss evolution: {0}'.format(losses))

    return torch.tensor([int(n[0]) for n in I], dtype=torch.int32, device=x.device), None # losses[-1]
