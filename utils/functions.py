from collections import defaultdict
import numpy as np
import torch
from sklearn.metrics import average_precision_score
import os

def _unique_sample(ids_dict, num):
    mask = np.zeros(num, dtype=np.bool)
    for _, indices in ids_dict.items():
        i = np.random.choice(indices)
        mask[i] = True
    return mask


def cmc(distmat, query_ids=None, gallery_ids=None,
        query_cams=None, gallery_cams=None, topk=100,
        separate_camera_set=False,
        single_gallery_shot=False,
        first_match_break=False):
    m, n = distmat.shape
    # Fill up default values
    if query_ids is None:
        query_ids = np.arange(m)
    if gallery_ids is None:
        gallery_ids = np.arange(n)
    if query_cams is None:
        query_cams = np.zeros(m).astype(np.int32)
    if gallery_cams is None:
        gallery_cams = np.ones(n).astype(np.int32)
    # Ensure numpy array
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_cams = np.asarray(query_cams)
    gallery_cams = np.asarray(gallery_cams)
    # distmat is a 2D matrix of size #query_photos X #gallery_photos
    # in case of market, this is 3368 X 15913
    # distmat[0][0] is the distance between query_photo[0] to gallery_photo[0]
    # indices of same shape is a sorted matrix from least 
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    # Compute CMC for each query
    ret = np.zeros(topk)
    num_valid_queries = 0
    for i in range(m):
        # Filter out the same id and same camera
        valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                 (gallery_cams[indices[i]] != query_cams[i]))
        if separate_camera_set:
            # Filter out samples from same camera
            valid &= (gallery_cams[indices[i]] != query_cams[i])
        if not np.any(matches[i, valid]):
            continue
        if single_gallery_shot:
            repeat = 10
            gids = gallery_ids[indices[i][valid]]
            inds = np.where(valid)[0]
            ids_dict = defaultdict(list)
            for j, x in zip(inds, gids):
                ids_dict[x].append(j)
        else:
            repeat = 1
        for _ in range(repeat):
            if single_gallery_shot:
                # Randomly choose one instance for each id
                sampled = (valid & _unique_sample(ids_dict, len(valid)))
                index = np.nonzero(matches[i, sampled])[0]
            else:
                index = np.nonzero(matches[i, valid])[0]
            delta = 1. / (len(index) * repeat)
            for j, k in enumerate(index):
                if k - j >= topk:
                    break
                if first_match_break:
                    ret[k - j] += 1
                    break
                ret[k - j] += delta
        num_valid_queries += 1
    if num_valid_queries == 0:
        raise RuntimeError("No valid query")
    return ret.cumsum() / num_valid_queries


def mean_ap(distmat, epoch, query_ids=None, gallery_ids=None,
            query_cams=None, gallery_cams=None):
    m, n = distmat.shape
    # Fill up default values
    if query_ids is None:
        query_ids = np.arange(m)
    if gallery_ids is None:
        gallery_ids = np.arange(n)
    if query_cams is None:
        query_cams = np.zeros(m).astype(np.int32)
    if gallery_cams is None:
        gallery_cams = np.ones(n).astype(np.int32)
    # Ensure numpy array
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_cams = np.asarray(query_cams)
    gallery_cams = np.asarray(gallery_cams)
    # Sort and find correct matches
    # indices[0] is the sorted
    # list of indices of images in the test set that are "most like"
    # AKA least distance, from image 0 in the query set
    # Image "0" being the image with label 0
    # for example, on market, after one epoch, indices[0][0] = 6192
    # Interpret that as saying image 0 in the query set corresponds
    # best, AKA has least distance to, image 6192 in the test set
    indices = np.argsort(distmat, axis=1)
    # the question then is, what id does test image 6192 have?
    # to get that, we must index our gallery_ids at indices[0][0]
    # and we find that it is id 1.
    # query_ids[:, np.newaxis]) makes query_ids from shape (m,) -> (m, 1)
    # this is same as doing query_ids.reshape(m,1)
    # here we are comparing, for each query image, how many of the predictions
    # match the actual id for tht query image.
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    # Compute AP for each query image
    aps = []
    # ------------ BELOW CODE FOR CSCE 625 ---------------
    # write out our predictions for unique id query images to a file
    if not os.path.exists("./predictions"):
        os.makedirs("./predictions")
    # keep track of last id bcs we only want to output to the file one entry for each unique id
    last_id = -1
    # ----------------------------------------------------
    for i in range(m):
        # Filter out the same id and same camera
        # first find the entries which are not the same id or not the same camera
        valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                 (gallery_cams[indices[i]] != query_cams[i]))
        # drop the entries in matches[i] which are the same id and same camera
        # y_true is the ground truth
        y_true = matches[i, valid]
        # yscore is the distance we said they were apart (normalized to [0, -1])
        y_score = -distmat[i][indices[i]][valid]
        # ... if they were all taken on the same camera, then don't include
        # this person in our results
        if not np.any(y_true):
            continue
        aps.append(average_precision_score(y_true, y_score))
        # ------------ BELOW CODE FOR CSCE 625 ---------------
        if query_ids[i] != last_id: # lets just look at different id's
            with open("./predictions/epoch{}.txt".format(epoch), "a") as f:
                # for the query picture, lets look at the id's we predicted for it!
                f.write("{}, {}, {}:{}\n".format(
                    i, query_ids[i], average_precision_score(y_true, y_score),
                    '|'.join(map(lambda x: '{}, {}, {}, {}'.format(*x),
                        zip(indices[i, valid][:10],
                            gallery_ids[indices][i, valid][:10],
                            y_score[:10],
                            y_true[:10].astype(int))))))
            last_id = query_ids[i]
    with open("./predictions/epoch{}.txt".format(epoch), "a") as f:
        f.write(str(np.mean(aps)))
    # ----------------------------------------------------
    if len(aps) == 0:
        raise RuntimeError("No valid query")
    return np.mean(aps)
