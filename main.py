import os
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from collections import Counter
import pandas as pd
from string import punctuation
import re
from nltk.corpus import stopwords
from pymystem3 import Mystem
punctuation += '–—«»…'
m = Mystem()


curr_dir = os.getcwd()
data_dir = os.path.join(curr_dir, 'friends-data')


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


all_episodes = []
for root, dirs, files in tqdm(os.walk(data_dir)):
    for name in files:
        with open(os.path.join(root, name), 'r', encoding='utf-8-sig') as f:
            episode = f.read()
            clean_episode = preprocessing(episode)
        all_episodes.append(clean_episode)

vectorizer = CountVectorizer(analyzer='word')
X = vectorizer.fit_transform(all_episodes)

matrix_freq = np.asarray(X.sum(axis=0)).ravel()
final_matrix = np.array([np.array(vectorizer.get_feature_names()), matrix_freq])

# вариант без pandas
sorted_freq = (sorted(
    zip(np.array(vectorizer.get_feature_names()),
        matrix_freq),
    key=lambda x: x[1],
    reverse=True))

# print(sorted_freq[0], sorted_freq[-1])

# скорее нельзя было перефитить векторайзер с min_df == количество документов, но по-моему изящно
vectorizer2 = CountVectorizer(analyzer='word', min_df=len(all_episodes))
X2 = vectorizer2.fit_transform(all_episodes)
final_matrix2 = np.array([np.array(vectorizer2.get_feature_names()), matrix_freq])
# print(vectorizer2.get_feature_names())


# реализация на pandas (больше работал с ним, умею пользоваться)
df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())

print('Words in all documents:')
for i in df[df != 0].dropna(axis=1).columns:
    print(i)
print()

c = Counter(df.sum().to_dict())
print('Most common word: ', c.most_common()[0])
print('Less common word: ', c.most_common()[-1])
print()

chars = [x.lower() for x in
         ['Моника', 'Мон', 'Рэйчел', 'Рейч', 'Чендлер', 'Чэндлер', 'Чен', 'Фиби', 'Фибс', 'Росс', 'Джоуи', 'Джо']]
print(dict(df[chars].sum().sort_values(ascending=False)))
print()

pop = list(df[chars].sum().sort_values(ascending=False).to_dict().items())
print('Most popular character: {} ({} words)'.format(pop[0][0].capitalize(), int(pop[0][1])))
input('press ENTER to exit')
