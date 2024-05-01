import numpy as np

# NOTE: false negatives are not considered by candidate generation models

# true positives in candidates
def true_positives(pred, actual):
    return len(np.intersect1d(pred, actual))

# false positives in candidates
def false_positives(pred, actual):
    return len(pred[~np.isin(pred, actual)])

# false negatives in candidates
def false_negatives(pred, actual):
    return len(actual) - true_positives(pred, actual)

# 1d average precision
def avg_precision(pred, actual):
    _tp = 0
    precision = 0.0

    if true_positives(pred, actual) == 0:
        return 0.0
    
    for i, cur in enumerate(actual):
        if cur in pred:
            _tp += 1
            precision += _tp / (i + 1)
    return precision / _tp

# 2d mean avg precision
def mean_avg_precision(pred, actual):
    precisions = []
    for cur_pred, cur_actual in zip(pred, actual):
        precisions.append(avg_precision(cur_pred, cur_actual))
    return np.mean(precisions)

# 2d precision at k
def precision_at_k(pred, actual, k):
    precision_sum = 0.0
    num_users = len(actual)
    
    for i in range(num_users):
        true_set = set(actual[i])
        pred_set = set(pred[i][:k])
        precision_sum += true_positives(pred_set, true_set) / float(len(true_set))
    
    precision_at_k = precision_sum / num_users
    return precision_at_k

# 1d average recall
def avg_recall(pred, actual):
    _tp = 0
    recall = 0.0

    if true_positives(pred, actual) == 0:
        return 0.0
    
    for i, cur in enumerate(pred):
        if cur in actual:
            _tp += 1
            recall += _tp / (i + 1)
    return recall / _tp

# 2d mean avg recall
def mean_avg_recall(pred, actual):
    recall = []
    for cur_pred, cur_actual in zip(pred, actual):
        recall.append(avg_recall(cur_pred, cur_actual))
    
    return np.mean(recall)

# 2d recall at k
def recall_at_k(pred, actual, k):
    recall_sum = 0.0
    num_users = len(actual)
    
    for i in range(num_users):
        true_set = set(actual[i])
        pred_set = set(pred[i][:k])
        recall_sum += true_positives(pred_set, true_set) / float(len(pred[i]))
    
    recall_at_k = recall_sum / num_users
    return recall_at_k