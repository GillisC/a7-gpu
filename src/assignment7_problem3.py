#!/usr/bin/env python3

import numpy as np
import cupy as cp
import argparse
import pandas as pd
import csv
import sys
import time

def batch_scan_matrix_mul(X, Q, b = None):
    """
    X: n*d dataset (in host memory)
    Q: m*d queries (in host memory)
    b: optional batch size
    Returns an m-vector of indices I; the value i reports the row in X such 
    that the Euclidean norm of ||X[I[i],:]-Q[i]|| is minimal

    X: Each row represents a vector in the dataset
    Q: Each row represents a query we want to issue and find the nearest neighbor of in X
    I: Each element represents "query <index> found closest neighbor to be <value> found.
    That value is the index into the dataset in for the closest vector.
    """
    m = Q.shape[0]
    I = np.zeros(m, dtype=np.int64) 
    
    if b is None:
        b = m

    Q_sq_sum = np.sum(Q ** 2, axis=1)
    X_sq_sum = np.sum(X ** 2, axis=1)

    X_sq_sum = X_sq_sum[:, np.newaxis]

    for i in range (0, m, b):
        queries = Q[i : i + b]
        queries_sq_sum = Q_sq_sum[i : i + b]
        batch_size = queries.shape[0]

        XQ_T = np.dot(X, queries.T)

        distances = X_sq_sum + queries_sq_sum - (2 * XQ_T)

        I[i : i + batch_size] = np.argmin(distances, axis=0)

    return I


def load_glove(fn):
    """
    Loads the glove dataset from the file
    Returns (X,L) where X is the dataset vectors and L is the words associated
    with the respective rows.
    """
    df = pd.read_table(fn, sep = ' ', index_col = 0, header = None,
                           quoting = csv.QUOTE_NONE, keep_default_na = False)
    X = np.ascontiguousarray(df, dtype = np.float32)
    L = df.index.tolist()
    return (X, L)

def load_pubs(fn):
    """
    Loads the pubs dataset from the file
    Returns (X,L) where X is the dataset vectors (easting,northing) and 
    L is the list of names of pubs, associated with each row
    """
    df = pd.read_csv(fn)
    L = df['name'].tolist()
    X = np.ascontiguousarray(df[['easting','northing']], dtype = np.float32)
    return (X, L)

def load_queries(fn):
    """
    Loads the m*d array of query vectors from the file
    """
    return np.loadtxt(fn, delimiter = ' ', dtype = np.float32)

def load_query_labels(fn):
    """
    Loads the m-long list of correct query labels from a file
    """
    with open(fn,'r') as f:
        return f.read().splitlines()

if __name__ == '__main__':
    parser = argparse.ArgumentParser( \
          description = 'Perform nearest neighbor queries under the '
          'Euclidean metric using linear scan, measure the time '
          'and optionally verify the correctness of the results')
    parser.add_argument(
        '-d', '--dataset', type=str, required=True,
        help = 'Dataset file (must be pubs or glove)')
    parser.add_argument(
        '-q', '--queries', type=str, required=True,
        help = 'Queries file (must be compatible with the dataset)')
    parser.add_argument(
        '-l', '--labels', type=str, required=False,
        help = 'Optional correct query labels; if provided, the correctness of returned results is checked')
    parser.add_argument(
        '-b', '--batch-size', type=int, required=False,
        help = 'Size of batches')
    args = parser.parse_args()

    t1 = time.time()
    if 'pubs' in args.dataset:
        (X,L) = load_pubs(args.dataset)
    elif 'glove' in args.dataset:
        (X,L) = load_glove(args.dataset)
    else:
        sys.stderr.write(f'{sys.argv[0]}: error: Only glove/pubs supported\n')
        exit(1)
    t2 = time.time()

    (n,d) = X.shape
    assert len(L) == n

    t3 = time.time()
    Q = load_queries(args.queries)
    t4 = time.time()

    assert X.flags['C_CONTIGUOUS']
    assert Q.flags['C_CONTIGUOUS']
    assert X.dtype == np.float32
    assert Q.dtype == np.float32
    
    m = Q.shape[0]
    assert Q.shape[1] == d

    t5 = time.time()
    QL = None
    if args.labels is not None:
        QL = load_query_labels(args.labels)
        assert len(QL) == m

    t6 = time.time()

    I = batch_scan_matrix_mul(X, Q, args.batch_size)

    t7 = time.time()

    assert I.shape == (m,)

    num_erroneous = 0
    if QL is not None:
        for (i,j) in enumerate(I):
            if QL[i] != L[j]:
                sys.stderr.write(f'{i}th query was erroneous: got "{L[j]}", '
                                     f'but expected "{QL[i]}"\n')
                num_erroneous += 1

    NN_time = t7 - t6
    print(f'Loading dataset ({n} vectors of length {d}) took', t2-t1)
    print(f'Performing {m} NN queries took', NN_time)
    print(f'Throughput', m / NN_time)
    print(f'Number of erroneous queries: {num_erroneous}')
