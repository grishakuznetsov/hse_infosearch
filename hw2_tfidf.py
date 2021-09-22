import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from string import punctuation
import re
from nltk.corpus import stopwords
from pymystem3 import Mystem
punctuation += '–—«»…'
m = Mystem()


np.set_printoptions(threshold=sys.maxsize)


# в TFIDFvectorizer же встроено l2 нормирование, поэтому не использовал функцию

# def normalize(x):
#    return x / np.linalg.norm(x)


def corpus_indexing(corpus):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer


def preprocessing(text):
    text = text.lower()
    text = text.replace('\n', ' ')
    text = re.sub(r'\d*', '', text)
    text = re.sub(r'[a-z][A-Z]*', '', text)
    text = re.sub(r'\\[a-z]*', '', text)
    for p in punctuation:
        text = text.replace(p, '')
    text = re.sub(r'\s{2,}', ' ', text)
    lemmas = m.lemmatize(text)
    text = ''.join(lemmas).split()
    clean = []
    for i, word in enumerate(text):
        if word not in stopwords.words('russian'):
            clean.append(word)
    clean = ' '.join(clean)
    return clean


def get_data(data_dir):
    all_episodes = []
    names = []
    for root, dirs, files in os.walk(data_dir):
        for name in files:
            names.append(name)
            with open(os.path.join(root, name),
                      'r', encoding='utf-8-sig') as f:
                episode = f.read()
                clean_episode = preprocessing(episode)
            all_episodes.append(clean_episode)
    return all_episodes, names


def query_processing(q, vec):
    q = preprocessing(q)
    q_vec = vec.transform([q]).toarray()
    return q_vec


def compute_similarity(q, docs):
    similarity = []
    for doc in docs:
        similarity.append(cosine_similarity(q, doc)[0][0])
    return np.array(similarity)


def search(query):
    corpus, names = get_data(friends_dir)
    X, vectorizer = corpus_indexing(corpus)  # индексация корпуса
    query_vec = query_processing(query, vectorizer)  # индексация запроса
    sim = compute_similarity(query_vec, X)  # подсчет cos sim
    sorted_idx = np.flip(np.argsort(sim))
    sorted_names = [names[x] for x in sorted_idx]
    return sorted_names


if __name__ == '__main__':
    curr_dir = os.getcwd()
    friends_dir = os.path.join(curr_dir, 'friends-data')
    query_input = input('Query: ')
    res = search(query_input)
    print(res)
