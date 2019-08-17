import _pickle as pickle
import pandas as pd
import numpy as np
import gensim
from fuzzywuzzy import fuzz
import nltk
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from tqdm import tqdm
from scipy.spatial.distance import cosine, euclidean
from nltk import word_tokenize
import re
from random import randint

stop_words = stopwords.words('english')
def clean_text(text):
    """ Pre process and convert texts to a list of words """
    text = str(text)
    text = text.lower()
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " is ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"[a-z]+\-[a-z]+", "", text)
    text = re.sub(r"[a-z]+\-", "", text)
    text = re.sub(r"\-[a-z]+", "", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    return text

model = gensim.models.KeyedVectors.load_word2vec_format('../data/GoogleNews-vectors-negative300.bin.gz', binary=True)
def wmd(s1, s2):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    stop_words = stopwords.words('english')
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return round(model.wmdistance(s1, s2), 3)
def sent2vec(s):
    words = str(s).lower()
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())

sel = list()
for i in range(40000):
    sel.append(randint(0, 400000))
data = pd.read_csv('questions.csv')
data = data.iloc[sel, :]
data.drop('qid1', axis=1, inplace=True)
data.drop("qid2", axis=1, inplace=True)
wh = ['where', 'why', 'what', 'who', 'whom', 'how', 'when', 'is', 'am', 'are', 'has', 'have', 'had', 'do', 'does','did']
for x in wh:
    if x in stop_words:
        stop_words.remove(x)

for s in data.head()['question1']:
    print(s, '\n')

data['question1'] = data.question1.apply(lambda x: clean_text(x))
data['question2'] = data.question2.apply(lambda x: clean_text(x))

for s in data.head()['question1']:
    print(s, '\n')

# Added Features.
data['word_overlap'] = [set(x[0].split()) & set(x[1].split()) for x in data[['question1', 'question2']].values]
data['common_word_cnt'] = data['word_overlap'].str.len()
data['text1_nostop'] = data['question1'].apply(lambda x: " ".join(x for x in x.split() if x not in stop_words))
data['text2_nostop'] = data['question2'].apply(lambda x: " ".join(x for x in x.split() if x not in stop_words))
data['word_overlap'] = [set(x[0].split()) & set(x[1].split()) for x in data[['text1_nostop', 'text2_nostop']].values]
data['common_nonstop_word_cnt'] = data['word_overlap'].str.len()
data['char_cnt_1'] = data['question1'].str.len()
data['char_cnt_2'] = data['question2'].str.len()
data['char_cnt_diff'] = (data['char_cnt_1'] - data['char_cnt_2']) ** 2
data['word_cnt_1'] = data['question1'].apply(lambda x: len(str(x).split()))
data['word_cnt_2'] = data['question2'].apply(lambda x: len(str(x).split()))
data['word_cnt_diff'] = (data['word_cnt_1'] - data['word_cnt_2']) ** 2
data['len_q1'] = data.question1.apply(lambda x: len(str(x)))
data['len_q2'] = data.question2.apply(lambda x: len(str(x)))
data['diff_len'] = (data.len_q1 - data.len_q2) ** 2
text1 = list(data['question1'])
text2 = list(data['question2'])
corpus1 = ' '.join(text1)
corpus2 = ' '.join(text2)
corpus = corpus1.lower() + corpus2.lower()
lem = WordNetLemmatizer()
corpus = lem.lemmatize(corpus, "v")
tags = pos_tag(corpus.split())
nouns = [i[0] for i in tags if i[1] in ("NN", "NNS", "NNP", "NNPS")]
def count_common_nouns(var1, var2, var3):
    count = 0
    for i in var1:
        if (i in var2) & (i in var3):
            count += 1
    return count
data['text1_lower'] = data['question1'].apply(lambda x: x.lower())
data['text2_lower'] = data['question2'].apply(lambda x: x.lower())
data['common_noun_cnt'] = [
    count_common_nouns(nltk.word_tokenize(lem.lemmatize(x[0], "v")), nltk.word_tokenize(lem.lemmatize(x[1], "v")),
                       nouns) for x in data[['question1', 'question2']].values]
# FUZZ WUZZ Features
data['fuzz_qratio'] = data.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)
data['fuzz_WRatio'] = data.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)
data['fuzz_partial_ratio'] = data.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)
data['fuzz_partial_token_set_ratio'] = data.apply(
    lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
data['fuzz_partial_token_sort_ratio'] = data.apply(
    lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
data['fuzz_token_set_ratio'] = data.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])),
                                          axis=1)
data['fuzz_token_sort_ratio'] = data.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])),
                                           axis=1)
data['wmd'] = data.apply(lambda x: wmd(x['question1'], x['question2']), axis=1)
question1_vectors = np.zeros((data.shape[0], 300))
error_count = 0
for i, q in tqdm(enumerate(data.question1.values)):
    question1_vectors[i, :] = sent2vec(q)
question2_vectors = np.zeros((data.shape[0], 300))
for i, q in tqdm(enumerate(data.question2.values)):
    question2_vectors[i, :] = sent2vec(q)
data['cosine_distance'] = [round(cosine(x, y), 3) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                                    np.nan_to_num(question2_vectors))]
data['euclidean_distance'] = [round(euclidean(x, y), 3) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                                          np.nan_to_num(question2_vectors))]

to_remove = ['word_overlap', 'text1_lower', 'text2_lower', 'question1', 'question2', 'id','text1_nostop','text2_nostop','word_cnt_1','word_cnt_2','char_cnt_1','char_cnt_2']
data = data.drop(to_remove, axis=1)
data.to_csv('./final_training_data.csv', index=False)
