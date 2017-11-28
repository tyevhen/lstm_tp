import nltk
import pandas as pd
import operator
import numpy as np
import string
from sklearn.feature_extraction.text import CountVectorizer
from tensorport import get_data_path

nltk.download()
# Read csv file and return Pandas DataFrame
def create_df(filename):
    df = pd.read_csv(filename)
    return df


# Delete punctuation and stopwords
def clean_text(text):
    stopwords = nltk.corpus.stopwords.words("english")
    lowercase_text = text.lower()
    tokenize_text = nltk.word_tokenize(lowercase_text)
    text_without_stopwords = [w for w in tokenize_text if w not in stopwords]
    pure_txt = [w for w in text_without_stopwords if w not in string.punctuation]
    return " ".join(pure_txt)


# Convert word to its normal form
def lemmatize_text(text, stem=False):
    """

    :param stem: If True - use stemming, False - use lemmatization. Defalut: lemmatizetion
    """
    text = text.split()
    if stem:
        stemmer = nltk.stem.PorterStemmer()
        normal_txt = [stemmer.stem(w) for w in text]
    else:
        lemmanizer = nltk.stem.WordNetLemmatizer()
        normal_txt = [lemmanizer.lemmatize(w) for w in text]
    return " ".join(normal_txt)


def get_vocabulary(df, length=3000, ngram_range=None):
    """

    :param length: total length of the vocabulary, least frequent words ignored
    :param ngram_range: tuple; if true - use ngram models, specified by this tuple example: (2, 2) - BiGram model
    :return: dict word:frequency
    """
    docs = df.text.values
    vec = CountVectorizer(max_features=length)
    if ngram_range:
        vec.ngram_range = ngram_range
    transformation = vec.fit_transform(docs)
    freq = np.ravel(transformation.sum(axis=0))

    vocab = dict(zip(vec.get_feature_names(), freq))
    return vocab


# Mapping for embeddings
def embedding_mapping(vocab):
    """

    :return: dict word: rank, where rank 2 means the most frequent word, rank 3 - less frequent and so on
             0, and 1 will be used for padding and unknown words (not in vocab)
    """
    sorted_vocab = sorted(vocab.items(), key=operator.itemgetter(1), reverse=True)
    sorted_words = [w for (w, f) in sorted_vocab]
    emb_map = {word: rank + 2 for rank, word in enumerate(sorted_words)}
    emb_map["PAD"] = 0
    emb_map["UNKNOWN"] = 1
    return emb_map


# Text to matrix, return list
def vectorize_sentence(sentence,  vocabulary, ngram_range=None):
    vectorizer = CountVectorizer(vocabulary=vocabulary)

    if ngram_range:
        vectorizer.ngram_range = ngram_range

    sentence_transform = vectorizer.fit_transform(sentence)
    return sentence_transform.toarray()


def sentence_to_emb(sentence, vocab, maxlen):
    sentence = sentence.split()

    if len(sentence) > maxlen:
        sentence = sentence[-maxlen:]
        out = []
    else:
        dif = maxlen - len(sentence)
        out = [0] * dif
    for word in sentence:
        out.append(vocab.get(word, 1))
    return np.array(out)

LOCAL_DATA_PATH = '~/tensorportdemo/'
train_data_path = get_data_path(
        dataset_name="yevhentysh/train-csv",
        local_root=LOCAL_DATA_PATH,
        local_repo='data',
        path="train")

train_df = create_df(train_data_path)

# train_df.text = train_df.text.apply(clean_text)
train_df.text = train_df.text.apply(lambda row: lemmatize_text(row))

vocab = get_vocabulary(train_df, length=5000)
emb = embedding_mapping(vocab)



