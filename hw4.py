import json
import numpy as np
from string import punctuation
import re
from tqdm.auto import tqdm
from nltk.corpus import stopwords
import pymorphy2
from functools import reduce
import operator
from scipy import sparse
from gensim.models import KeyedVectors
import torch
import pickle
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

morph = pymorphy2.MorphAnalyzer()
punctuation += '–—«»…'

bert_tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")
bert_model = AutoModel.from_pretrained("cointegrated/rubert-tiny")

ft_model = KeyedVectors.load('araneum_none_fasttextcbow_300_5_2018.model')


count_vectorizer = CountVectorizer()
tf_vectorizer = TfidfVectorizer(use_idf=False, norm='l2')
tfidf_vectorizer = TfidfVectorizer(use_idf=True, norm='l2')


def norm(x):
    return x / np.linalg.norm(x)


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


def get_data(filename):  # загрузка данных - переделать
    with open(filename, 'r', encoding='utf-8') as f:
        corpus = list(f)[:10000]
    all_questions = list()
    all_answers = list()
    for i in tqdm(range(len(corpus))):
        sample = json.loads(corpus[i])
        if sample['answers']:
            texts = list()
            values = list()
            for answer in sample['answers']:
                if answer['author_rating']['value']:
                    texts.append(answer['text'])
                    values.append(int(answer['author_rating']['value']))
            all_answers.append(texts[np.argsort(values)[-1]])
            all_questions.append(' '.join(sample['question']) + ' '.join(sample['comment']))
    return all_questions, all_answers


def embed_bert_cls(text, model, tokenizer):
    t = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**{k: v.to(model.device) for k, v in t.items()})
    embeddings = model_output.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings[0].cpu().numpy()


def tf_idf_indexing(docs):
    return tfidf_vectorizer.fit_transform(docs)


def cv_indexing(docs):
    return count_vectorizer.fit_transform(docs)


def bm25_indexing(docs):
    x_count_vec = count_vectorizer.fit_transform(docs)
    x_tf_vec = tf_vectorizer.fit_transform(docs)
    x_tfidf_vec = tfidf_vectorizer.fit_transform(docs)
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


def bert_embeddings(docs):
    embs = []
    for doc in tqdm(docs):
        embs.append(embed_bert_cls(doc, bert_model, bert_tokenizer))
    return sparse.csr_matrix(embs)


def ft_embeddings(docs):
    embs = []
    for text in docs:
        words = text.split()
        token_embs = np.zeros((len(words), ft_model.vector_size))
        for i, word in enumerate(words):
            token_embs[i] = ft_model[word]
        if token_embs.shape[0] != 0:
            embs.append(norm(np.mean(token_embs, axis=0)))
    return sparse.csr_matrix(embs)


def search(corpus, embeddings, query):
    scores = np.dot(embeddings, query.T).toarray()
    sorted_scores_indx = np.argsort(scores, axis=0)[::-1]
    corpus = np.array(corpus)[sorted_scores_indx.ravel()]
    return corpus



def bert_query_proc(query):
    return sparse.csr_matrix(embed_bert_cls([query], bert_model, bert_tokenizer))


def ft_query_proc(query):
    return ft_embeddings(query)


def tf_idf_query_proc(query):
    return tfidf_vectorizer.transform(query)


def cv_query_proc(query):
    return count_vectorizer.transform(query)


def bm25_query_proc(query):
    return tfidf_vectorizer.transform(query)


def main():
    questions, answers = get_data('questions_about_love.jsonl')
    query = 'Что делать?'
    clean_answers = [preprocessing(x) for x in tqdm(answers)]
    clean_questions = [preprocessing(x) for x in tqdm(questions)]
    clean_query = preprocessing(query)

    bert_answers = bert_embeddings(answers)
    bert_questions = bert_embeddings(questions)
    bert_query = bert_query_proc(query)
    result = search(answers, bert_answers, bert_query)

    sparse.save_npz('indexing/BERT_answers.npz', bert_answers)
    sparse.save_npz('indexing/BERT_questions.npz', bert_questions)

    print('BERT')
    for i in range(5):
        print(result[i])
    print()

    ft_answers = ft_embeddings(clean_answers)
    ft_questions = ft_query_proc(clean_questions)
    ft_query = ft_query_proc(clean_query)
    result = search(answers, ft_answers, ft_query)

    sparse.save_npz('indexing/FASTTEXT_answers.npz', ft_answers)
    sparse.save_npz('indexing/FASTTEXT_questions.npz', ft_questions)

    print('FASTTEXT')
    for i in range(5):
        print(result[i])
    print()

    tf_ifd_answers = tf_idf_indexing(clean_answers)
    tf_ifd_questions = tf_idf_query_proc(clean_questions)
    tf_idf_query = tf_idf_query_proc([clean_query])
    result = search(answers, tf_ifd_answers, tf_idf_query)

    sparse.save_npz('indexing/TF-IDF_answers.npz', tf_ifd_answers)
    sparse.save_npz('indexing/TF-IDF_questions.npz', tf_ifd_questions)
    pickle.dump(tf_vectorizer, open('vectorizers/TF-IDF_vec.pickle', 'wb'))

    print('TF-IDF')
    for i in range(5):
        print(result[i])
    print()

    cv_answers = cv_indexing(clean_answers)
    cv_questions = cv_query_proc(clean_questions)
    cv_query = cv_query_proc([clean_query])
    result = search(answers, cv_answers, cv_query)

    sparse.save_npz('indexing/CV_answers.npz', cv_answers)
    sparse.save_npz('indexing/CV_questions.npz', cv_questions)
    pickle.dump(count_vectorizer, open('vectorizers/CV_vec.pickle', 'wb'))

    print('CV')
    for i in range(5):
        print(result[i])
    print()

    bm25_answers = bm25_indexing(clean_answers)
    bm25_questions = bm25_query_proc(clean_questions)
    bm25_query = bm25_query_proc([clean_query])
    result = search(answers, bm25_answers, bm25_query)

    sparse.save_npz('indexing/BM25_answers.npz', bm25_answers)
    sparse.save_npz('indexing/BM25_questions.npz', bm25_questions)
    pickle.dump(tf_vectorizer, open('vectorizers/BM25_vec.pickle', 'wb'))

    print('BM25')
    for i in range(5):
        print(result[i])


if __name__ == '__main__':
    main()
