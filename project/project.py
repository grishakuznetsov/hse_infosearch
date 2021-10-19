import streamlit as st
import json
import numpy as np
from string import punctuation
import re
from tqdm.auto import tqdm
from nltk.corpus import stopwords
import pymorphy2
from scipy import sparse
from gensim.models import KeyedVectors
import torch
import time
import pickle
from transformers import AutoTokenizer, AutoModel

morph = pymorphy2.MorphAnalyzer()
punctuation += '–—«»…'


bert_index = sparse.load_npz('indexing/BERT_answers.npz')
ft_index = sparse.load_npz('indexing/FASTTEXT_answers.npz')
bm25_index = sparse.load_npz('indexing/BM25_answers.npz')
cv_index = sparse.load_npz('indexing/CV_answers.npz')
tfidf_index = sparse.load_npz('indexing/TF-IDF_answers.npz')
count_vectorizer = pickle.load(open('vectorizers/CV_vec.pickle', 'rb'))
tfidf_vectorizer = pickle.load(open('vectorizers/TF-IDF_vec.pickle', 'rb'))
bm25_vectorizer = pickle.load(open('vectorizers/BM25_vec.pickle', 'rb'))


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


def ft_embeddings(docs):
    embs = []
    for text in docs:
        words = text.split()
        token_embs = np.zeros((len(words), ft_model.vector_size))
        for i, word in enumerate(words):
            token_embs[i] = ft_model[word]
        if token_embs.shape[0] != 0:
            emb = norm(np.mean(token_embs, axis=0))
        embs.append(emb)
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


@st.cache(allow_output_mutation=True)
def load_models():
    bert_model = AutoModel.from_pretrained("cointegrated/rubert-tiny")
    bert_tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")
    ft_model = KeyedVectors.load('araneum_none_fasttextcbow_300_5_2018.model')
    return bert_model, bert_tokenizer, ft_model


def choose(mode):
    if mode == 'BERT':
        vec = None
        return bert_index, bert_query_proc, vec

    elif mode == 'FASTTEXT':
        vec = None
        return ft_index, ft_query_proc, vec

    elif mode == 'BM25':
        return bm25_index, bm25_query_proc, tfidf_vectorizer

    elif mode == 'CV':

        return cv_index, cv_query_proc, count_vectorizer

    elif mode == 'TF-IDF':
        return tfidf_index, tf_idf_query_proc, tfidf_vectorizer


bert_model, bert_tokenizer, ft_model = load_models()


def main():
    col1, col2 = st.columns(2)
    with col1:
        st.image('https://acegif.com/wp-content/uploads/gif-i-love-you-69.gif')
    with col2:
        st.image('index.jpg')
    st.video('https://youtu.be/tg00YEETFzg?t=88')
    st.title('Всё, что вы всегда хотели знать о любви, но боялись спросить')
    _, corpus = get_data('questions_about_love.jsonl')
    options = ['BERT', 'BM25', 'CV', 'FASTTEXT', 'TF-IDF']
    mode = st.selectbox(label='Метод', options=options)
    out_num = st.slider('Количество ответов', 0, 10, 5)
    query = st.text_input('Найти!')

    start = time.time()
    index, query_proc, vec = choose(mode)
    if mode != 'BERT':
        query = preprocessing(query)
    if vec is None:
        query = query_proc(query)
        res = search(corpus, index, query)
        for i in range(out_num):
            st.write(f'{i+1}. {res[i]}')
        st.write(time.time() - start)
    else:
        query = query_proc([query])
        res = search(corpus, index, query)
        for i in range(out_num):
            st.write(f'{i+1}. {res[i]}')
        st.write(time.time() - start)


if __name__ == '__main__':
    main()

