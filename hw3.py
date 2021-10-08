import json
import operator
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from string import punctuation
import re
from tqdm.auto import tqdm
from nltk.corpus import stopwords
import pymorphy2
from functools import reduce
from scipy import sparse

morph = pymorphy2.MorphAnalyzer()
punctuation += '–—«»…'


count_vectorizer = CountVectorizer()
tf_vectorizer = TfidfVectorizer(use_idf=False, norm='l2')
tfidf_vectorizer = TfidfVectorizer(use_idf=True, norm='l2')

# Заменил mystem на pymorphy потому что быстрее (нанмого)


def preprocessing(text):
    text = text.lower()
    text = text.replace('\n', ' ')
    text = re.sub(r'\d*', '', text)
    text = re.sub(r'[a-z][A-Z]*', '', text)
    text = re.sub(r'\\[a-z]*', '', text)
    for p in punctuation:
        text = text.replace(p, '')
    text = re.sub(r'\s{2,}', ' ', text).split()
    text = [morph.parse(x)[0].normal_form for x in text]
    clean = list()
    for i, word in enumerate(text):
        if word not in stopwords.words('russian'):
            clean.append(word)
    clean = ' '.join(clean)
    return clean


def get_data(filename):  # загрузка данных
    with open(filename, 'r', encoding='utf-8') as f:
        corpus = list(f)[:50000]
    all_texts = list()
    for i in tqdm(range(len(corpus))):
        sample = json.loads(corpus[i])
        if sample['answers']:
            texts = list()
            values = list()
            for answer in sample['answers']:
                if answer['author_rating']['value']:
                    texts.append(answer['text'])
                    values.append(int(answer['author_rating']['value']))
            all_texts.append(texts[np.argsort(values)[-1]])
    return all_texts


def corpus_indexing(texts):  # индексация корпуса
    x_count_vec = count_vectorizer.fit_transform(texts)
    x_tf_vec = tf_vectorizer.fit_transform(texts)
    x_tfidf_vec = tfidf_vectorizer.fit_transform(texts)
    idf = tfidf_vectorizer.idf_
    idf = np.expand_dims(idf, axis=0)
    tf = x_tf_vec

    k = 2
    b = 0.75
    len_d = x_count_vec.sum(axis=1)
    avdl = len_d.mean()

    B_1 = (k * (1 - b + b * len_d / avdl))
    B_1 = np.expand_dims(B_1, axis=-1)
    rows, cols, values = list(), list(), list()
    for i, j in zip(*tf.nonzero()):
        rows.append(i)
        cols.append(j)
        A = reduce(operator.mul, [idf[0][j], tf[i, j], (k + 1)])
        B = tf[i, j] + B_1[i]
        values.append((A / B)[0][0])
    return sparse.csr_matrix((values, (rows, cols)))


def get_query(query):  # обработка запроса
    return tfidf_vectorizer.transform([preprocessing(query)])


def my_otvety_mailru(corpus, query_vec, corpus_matrix, n_answers):  # функция поиска ответов
    scores = np.dot(corpus_matrix, query_vec.T).toarray()
    sorted_scores_indx = np.argsort(scores, axis=0)[::-1]
    top = np.array(corpus)[sorted_scores_indx.ravel()[:int(n_answers)]]
    return top


def main():  # локомотив для всего остального
    corpus_data = get_data('questions_about_love.jsonl')
    clean_corpus_data = [preprocessing(x) for x in tqdm(corpus_data)]
    corpus_matrix = corpus_indexing(clean_corpus_data)
    question = input('Question: ')
    num_answers = input('How many answers: ')
    query = get_query(question)
    query_answers = my_otvety_mailru(corpus_data,
                                     query,
                                     corpus_matrix,
                                     num_answers)
    for answer in query_answers:
        print('\t'+answer)


if __name__ == '__main__':
    main()
