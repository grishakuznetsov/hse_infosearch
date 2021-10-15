import numpy as np
from scipy import sparse

import os
from tqdm import tqdm


def sort_query(index, query):
    scores = np.dot(index, query.T).toarray()
    return np.argsort(scores, axis=0)[::-1, :]


def evaluate(sorted_scores_indx):
    score = 0
    for index, row in enumerate(sorted_scores_indx):
        top = row[:5]
        if index in top:
            score += 1
    return score / len(sorted_scores_indx)


def load_indexes():
    indexes = list()
    questions = list()
    for file in os.listdir('indexing'):
        if 'answers' in file:
            indexes.append(file)
        else:
            questions.append(file)
    return indexes, questions


def main():
    indexes, questions = load_indexes()
    metrics = []
    for i in tqdm(range(len(indexes))):
        index = sparse.load_npz(f'indexing/{indexes[i]}')
        query = sparse.load_npz(f'indexing/{questions[i]}')
        sorted_scores_indx = sort_query(index, query)
        metrics.append(evaluate(sorted_scores_indx))

    index_names = [x.split('_')[0] for x in indexes]
    for i in range(len(indexes)):
        print(f'Index: {index_names[i]} \n Score: {metrics[i]}')


if __name__ == '__main__':
    main()
