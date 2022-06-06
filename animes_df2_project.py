#%% importing packages
import pandas as pd
import contractions

#import nltk
#from nltk import sent_tokenize
from nltk import word_tokenize
from nltk.corpus import stopwords

#import os
#import csv
#import sys

import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#from sklearn.pipeline import Pipeline, make_pipeline
#from sklearn.datasets import make_regression, make_classification
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import confusion_matrix, precision_score, precision_recall_curve, recall_score, f1_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from matplotlib import pyplot as plt
import matplotlib.cm as cm
import plotly.graph_objects as go

import gensim
import gensim.models
from gensim.models import word2vec, FastText

import re
import numpy as np

# https://pypi.org/project/pyenchant/   # help(enchant)
#help(enchant)

import eli5


stop_words = set(stopwords.words('english'))



#%% #def customfeats
class CustomFeats(BaseEstimator, TransformerMixin):
    def __init__(self):
      self.feat_names = set()

    def fit(self, x, y=None):
        return self

    @staticmethod
    def features(review):
      return {
          'bias' : 4.0,
          'RAM' : test_binary_feature(review),
          'mac': mac_binary_feature(review),
          'apple': apple_binary_feature(review),
          'IBM': ibm_feature(review)
      }

    def get_feature_names(self):
        return list(self.feat_names)
      
    def transform(self, reviews):
      feats = []
      for review in reviews:
        f = self.features(review)
        [self.feat_names.add(k) for k in f] 
        feats.append(f)
      return feats
#%% def text_cleaner(text)

def text_cleaner(text):
    #text = text.lower()
    text = re.sub(string = text, pattern = r'[^A-Za-z0-9\-\_\=\+\!\@\#\$\%\^\&\*\(\)\,\.\/\<\>\?\'\;\:\"\{\}`~\[\] ]', repl = " ")
    text = re.sub(r'--',' ',text)
    text = re.sub(pattern = "\[.*?\]", repl = "", string = text)
    text = re.sub(string = text, pattern = r'\( ?(source|Source):? *[A-Za-z0-9\w\W ]+\)', repl = "")
    text = re.sub(string = text, pattern = r"(\b|\s+\-?|^\-?)(\d+|\d*\.\d+)\b", repl = " ")   
    text = re.sub(string = text, pattern = r'\!|\?|\.|\,|\:|\;|\"|\*|\[|\]|\(|\)|\/|~|`', repl = " ")
    text = re.sub(string = text, pattern = r' +', repl = " ")
    text = ' '.join(text.split())
    return text

#%% def filtered_synopsis_frame(animes, string_column_name) - cleaning the synopsis text and removing stopwords, returns list of synopsis (untokenized by word), a list of unused words for checking, and a counter detailing the total number of words that were removed
def filtered_synopsis_frame(animes, string_column_name):
    dic = []
    d = enchant.Dict("en_US")   #importing a dictionary so that I can check whether or not words are in the contents and use that to remove names
    counter = 0
    unused_words = []
    
    for x in animes[str(string_column_name)]:
        #p = text_cleaner(x)
        words = word_tokenize(x)
        
        
        
       
        filtered = []
        for w in words:
            if len(w) <= 2:
                continue
            
            if w in stop_words:
                counter += 1
                continue
            
            if w in unused_words:
                counter += 1
                continue
            
            if d.check(w) == 0:
                unused_words.append(w)
                counter += 1
            
            if d.check(w) == 1:
                filtered.append(w)
            
        j = " ".join(filtered)        
        dic.append(j)
        
    new_df = animes
    new_df[str(string_column_name)] = dic
        
    return new_df, unused_words, counter   #modified
    #return dic, unused_words, counter  #original
    
#%% def repl_special_chars(text) - function for lowercasing all text, cleaning out all accented letters and replacing them with unaccented letters
def repl_special_chars(text):
    text = text.lower()
    #text = re.sub(string = text, pattern = r'á|à|ȧ|â|ä|ǎ|ă|ā|ã|å|ą|ⱥ|ấ|ầ|ắ|ằ|ǡ|ǻ||ǟ|ẫ|ẵ|ả|ȁ|ȃ|ẩ|ẳ|ạ|ḁ|ậ|ặ|æ|ǽ|ǣ', repl = "a")
    text = re.sub(string = text, pattern = r'é|è|ė|ê|ë|ě|ĕ|ē|ẽ|ę|ȩ|ɇ|ế|ề|ḗ|ḕ|ễ|ḝ|ẻ|ȅ|ȇ|ể|ẹ|ḙ|ḛ|ệ', repl = "e")
    text = re.sub(string = text, pattern = r"ı|í|ì|î|ï|ǐ|ĭ|ī|ĩ|į|ɨ|ḯ|ỉ|ȉ|ȋ|ị|ḭ|ĳ", repl = "i")
    text = re.sub(string = text, pattern = r"ó|ò|ȯ|ô|ö|ǒ|ŏ|ō|õ|ǫ|ő|ố|ồ|ø|ṓ|ṑ|ȱ|ṍ|ȫ|ỗ|ṏ|ǿ|ȭ|ǭ|ỏ|ȍ|ȏ|ơ|ổ|ọ|ớ|ờ|ỡ|ộ|ƣ|ở|ợ|œ", repl = "o")
    text = re.sub(string = text, pattern = r"ú|ù|û|ü|ǔ|ŭ|ū|ũ|ů|ų|ű|ʉ|ǘ|ǜ|ǚ|ṹ|ǖ|ṻ|ủ|ȕ|ȗ|ư|ụ|ṳ|ứ|ừ|ṷ|ṵ|ữ|ử|ự", repl = "u")
    
    #text = re.sub(string = text, pattern = "ï", repl = "i")
    
    
    
    return text

#%% def do_plot

def do_plot(X_fit, labels):
  dimension = X_fit.shape[1]
  label_types = list(set(labels))
  num_labels = len(list(set(labels)))
  colors = cm.brg(np.linspace(0, 1, num_labels))
  if num_labels == X_fit.shape[0]:
    label_types = sorted(label_types, key=lambda k: np.where(labels==k))
    colors = cm.seismic(np.linspace(0, 1, num_labels))
  if dimension == 2:
    for lab, col in zip(label_types, colors):
      plt.scatter(X_fit[labels==lab, 0],
                  X_fit[labels==lab, 1],
                  label=lab,
                  c=col, alpha=0.5)
  else:
    raise Exception('Unknown dimension: %d' % dimension)
  plt.legend(loc='best')
  plt.show()
  
  #%% def split_dataset - splittings dataset into test and train for vectorizer (as shown in lecture notes)
  class Dataset:
    def __init__(self, dataset, start_idx, end_idx):
      self.data = dataset.data[start_idx:end_idx]
      self.labels = dataset.target[start_idx:end_idx]
      self.vecs = None

      
  def split_dataset(dataset, train_rate=0.8):
    data_size = len(dataset.data)
    train_last_idx = int(train_rate * data_size)
    train = Dataset(dataset, 0, train_last_idx)
    test = Dataset(dataset, train_last_idx, data_size)
    return train, test 

#%% def word2idx  and idx2word - word2idx and idx2word functions
def word2idx(word, vocab_dict):
  index = vocab_dict[word] if word in vocab_dict.keys() else 'Not Found'
  print(word, ' -> ', index)


def idx2word(index, vocabs):
  word = vocabs[index] if 0 <= index < len(vocabs) else 'Not Found'
  print(index, ' -> ', word)
  
#%% def reduce dimensions, plot_with_plotly, visualizing embeddings 

from sklearn.decomposition import IncrementalPCA    # inital reduction
from sklearn.manifold import TSNE                   # final reduction
import numpy as np                                  # array handling


def reduce_dimensions(model):
    num_dimensions = 2  # final num dimensions (2D, 3D, etc)

    # extract the words & their vectors, as numpy arrays
    vectors = np.asarray(model.wv.vectors)
    labels = np.asarray(model.wv.index_to_key)  # fixed-width numpy strings

    # reduce using t-SNE
    tsne = TSNE(n_components = num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return x_vals, y_vals, labels




def plot_with_plotly(x_vals, y_vals, labels, plot_in_notebook = False):
    from plotly.offline import init_notebook_mode, iplot, plot
    import plotly.graph_objs as go

    trace = go.Scatter(x=x_vals, y=y_vals, mode='text', text=labels)
    data = [trace]

    if plot_in_notebook:
        init_notebook_mode(connected=True)
        iplot(data, filename='word-embedding-plot')
    else:
        plot(data, filename='word-embedding-plot.html')


def plot_with_matplotlib(x_vals, y_vals, labels):
    import matplotlib.pyplot as plt
    import random

    random.seed(0)

    plt.figure(figsize=(12, 12))
    plt.scatter(x_vals, y_vals)

    #
    # Label randomly subsampled 25 data points
    #
    indices = list(range(len(labels)))
    selected_indices = random.sample(indices, 25)
    for i in selected_indices:
        plt.annotate(labels[i], (x_vals[i], y_vals[i]))

try:
    get_ipython()
except Exception:
    plot_function = plot_with_matplotlib
else:
    plot_function = plot_with_plotly


#%% def classification
def classification(vectorizer, model, fit_vect=False):
  if fit_vect:
    vectorizer.fit(train.data)
  train.vecs = vectorizer.transform(train.data)
  test.vecs = vectorizer.transform(test.data)
  model.fit(train.vecs, train.labels)
  train_preds = model.predict(train.vecs)
  train_f1 = f1_score(train.labels, train_preds, average='micro')
  test_preds = model.predict(test.vecs)
  test_f1 = f1_score(test.labels, test_preds, average='micro')
  return train_f1, test_f1  

#%% reading in data
#desktop
animes2 = pd.read_csv("C:/Users/chaff/Desktop/Boring Adult Stuff/School Work/UC Irvine/Spring Quarter/NLP/Project/animes_df2.csv", header = 0)

#laptop
#animes2 = pd.read_csv("C:/Users/chaff/Desktop/UC Irvine/Spring Quarter/NLP/Project/animes_df2.csv", header = 0)

drop_labels2 = ['Unnamed: 0']
animes2.drop(axis = 1, labels = drop_labels2, inplace = True)

animes2.dropna(inplace = True)


adults = animes2.loc[animes2['adult'] == 1]
kids = animes2.loc[animes2['adult'] == 0]

adult_train, adult_test = sklearn.model_selection.train_test_split(adults, train_size = 0.8, test_size = 0.2, random_state = 42069)
kids_train, kids_test = sklearn.model_selection.train_test_split(kids, train_size = 0.8, test_size = 0.2, random_state = 42069)

train = adult_train.append(kids_train)
test = adult_test.append(kids_test)

#len(train['data'])
#len(test['data'])

drop_labels3 = ['genre','title']
train.drop(axis = 1, labels = drop_labels3, inplace = True)
test.drop(axis = 1, labels = drop_labels3, inplace = True)

train.columns = ['data', 'target']
test.columns = ['data', 'target']


#%% Embeddings model that has everything (both adult and kid content)
all_sentences = []


for x in train['data']:
    y = word_tokenize(x)
    all_sentences.append(y)

for x in test['data']:
    y = word_tokenize(x)
    all_sentences.append(y)

all_model = gensim.models.Word2Vec(sentences = all_sentences, min_count = 4, epochs = 1000)
#vec_king = all_model.wv['sister']

for index, word in enumerate(all_model.wv.index_to_key):
    print(f"word #{index}/{len(all_model.wv.index_to_key)} is {word}")
    
    #unique vocab count =  9204

all_model.build_vocab(all_sentences, progress_per = 1000)

all_model.train(all_sentences, total_examples = all_model.corpus_count, epochs = all_model.epochs)

#laptop
#all_model = gensim.models.Word2Vec.load("C:/Users/chaff/Desktop/UC Irvine/Spring Quarter/NLP/Project/all_animes_train_model.model") # laptop

#desktop
all_model = gensim.models.Word2Vec.load("C:/Users/chaff/Desktop/Boring Adult Stuff/School Work/UC Irvine/Spring Quarter/NLP/Project/all_animes_train_model.model") #desktop
#all_model.save("C:/Users/chaff/Desktop/Boring Adult Stuff/School Work/UC Irvine/Spring Quarter/NLP/Project/all_animes_train_model2.model")

all_model.wv.most_similar("monster")  # really good example

#test = all_model.wv.most_similar("boy", topn = 10)


#comparison of similarity between two words
all_model.wv.similarity(w1 = "knife", w2 = "sword")


#print 5 words most similar to any of the parameters passed
print(all_model.wv.most_similar(positive=['sword', 'knife'], topn=5))

#all_model.wv.most_similar_cosmul(positive = ['class'], negative = ['adult'], topn=5)
#all_model.wv.most_similar_cosmul(positive = ['friend','enemy'],  topn=5)
all_model.wv.most_similar_cosmul(positive = ['devil'], negative = ['lord'], topn=5)

x_vals, y_vals, labels = reduce_dimensions(all_model)

plot_function(x_vals, y_vals, labels)

#%%
word_vectors = all_model.wv


#%% word plot
words = np.array(['sword', 'artifact', 'school', 'hero', 'save', 'legendary', 'magic', 'relic', 'world', 'princess', 'tennis','tea'])
factors = TruncatedSVD(2).fit_transform(word_vectors[words])

# do_plot(, words)
factors.shape
do_plot(factors, words)

#%% word plot 2
words = np.array(['club', 'basketball', 'soccer', 'tennis', 'volleyball', 'team', 'sports', 'rival', 'school', 'monster', 'salary','work'])
factors = TruncatedSVD(2).fit_transform(word_vectors[words])

# do_plot(, words)
factors.shape
do_plot(factors, words)

#%% KNN W2V (this ones not great)
num_neighs = 10
metric = 'cosine'

synopsis_list_train = []
synopsis_list_test = []

for x in train.data:
    z = word_tokenize(x)
    synopsis_list_train.append(z)
    
for x in test.data:
    z = word_tokenize(x)
    synopsis_list_test.append(z)

#train.w2v = np.zeros((len(train.data), word_vectors['good'].shape[0])) #original from notes
#test.w2v = np.zeros((len(test.data), word_vectors['good'].shape[0])) #original from notes

train.w2v = np.zeros((len(train.data), word_vectors['father'].shape[0])) #experimental
test.w2v = np.zeros((len(test.data), word_vectors['father'].shape[0])) #experimental (why is the word good here?)

nbrs_w2v = NearestNeighbors(n_neighbors = num_neighs, algorithm='brute', metric = metric).fit(train.w2v)
nbrs_w2v_classifier = KNeighborsClassifier(num_neighs).fit(train.w2v, train.target)
print('w2v accuracy is', accuracy_score(test.target, nbrs_w2v_classifier.predict(test.w2v)))

#original
#w2v accuracy is 0.3791848617176128

#1000 epochs
#w2v accuracy is 0.3791848617176128

#%% KNN vecs classification (features created with TFIDF)
num_neighs = 10
metric = 'cosine'

features = TfidfVectorizer(lowercase=True, stop_words='english', min_df=2, max_df=0.5, ngram_range = (1,2))
train.vecs = features.fit_transform(train.data)
test.vecs = features.transform(test.data)

  
nbrs_vecs = NearestNeighbors(n_neighbors=num_neighs, algorithm='brute', metric=metric).fit(train.vecs)
nbrs_vecs_classifier = KNeighborsClassifier(num_neighs).fit(train.vecs, train.target)
print('Train KNN TFIDF vecs accuracy is', accuracy_score(train.target, nbrs_vecs_classifier.predict(train.vecs)))
print('Test KNN TFIDF vecs accuracy is', accuracy_score(test.target, nbrs_vecs_classifier.predict(test.vecs)))

#TFIDF train vecs accuracy is 0.6847133757961783
#TFIDF test vecs accuracy is 0.5862445414847162

#%% KNN vecs classification (features created with CountVectorizer)
num_neighs = 10
metric = 'cosine'

features = CountVectorizer(lowercase=True, stop_words='english', min_df=2, ngram_range = (1,2))
train.vecs = features.fit_transform(train.data)
test.vecs = features.transform(test.data)

  
nbrs_vecs = NearestNeighbors(n_neighbors=num_neighs, algorithm='brute', metric=metric).fit(train.vecs)
nbrs_vecs_classifier = KNeighborsClassifier(num_neighs).fit(train.vecs, train.target)
print('Train KNN BoW vecs accuracy is', accuracy_score(train.target, nbrs_vecs_classifier.predict(train.vecs)))
print('Test KNN BoW vecs accuracy is', accuracy_score(test.target, nbrs_vecs_classifier.predict(test.vecs)))

#Train KNN BoW vecs accuracy is 0.6626933575978162
#Test KNN BoW vecs accuracy is 0.5549490538573508

#%% KNN SVD vecs (features gained using TFIDF) 

num_neighs = 10
metric = 'cosine'

features = TfidfVectorizer(lowercase=True, stop_words = 'english', min_df=2, max_df = 0.8, ngram_range = (1,2), max_features = 25) 
train.vecs = features.fit_transform(train.data)
test.vecs = features.transform(test.data)

#factors = TruncatedSVD(300).fit_transform(train.vecs.toarray())
#do_plot(factors)

svd_pass =  [300,200,100,1]
train_acc = []
test_acc = []

for x in svd_pass:
    
    svd = TruncatedSVD(x)

    train.svd = svd.fit_transform(train.vecs.toarray())
    test.svd = svd.transform(test.vecs.toarray())
    train.svd.shape, test.svd.shape
    
    nbrs_svd = NearestNeighbors(n_neighbors=num_neighs, algorithm='brute', metric=metric).fit(train.svd)
    nbrs_svd_classifier = KNeighborsClassifier(num_neighs).fit(train.svd, train.target)
    
    #print('#Train svd accuracy is ', accuracy_score(train.target, nbrs_svd_classifier.predict(train.svd))," @ ", x, "SVD", "(using TFIDF)")
    #print('#Test svd accuracy is ', accuracy_score(test.target, nbrs_svd_classifier.predict(test.svd)), " @ ", x, "SVD", "(using TFIDF)")
    x = accuracy_score(train.target, nbrs_svd_classifier.predict(train.svd))
    train_acc.append(x)
    
    y = accuracy_score(test.target, nbrs_svd_classifier.predict(test.svd))
    test_acc.append(y)
    
counter = 0
for x in train_acc:
    print('#Train svd accuracy is ', x, " @ ", svd_pass[int(counter)], "SVD", "(using CountVectorizer)")
    counter += 1

counter = 0
for x in test_acc:
    print('#Test svd accuracy is ', x, " @ ", svd_pass[int(counter)], "SVD", "(using CountVectorizer)")
    counter += 1
    
#for x in svd_pass:
    
#    svd = TruncatedSVD(x)

#    train.svd = svd.fit_transform(train.vecs.toarray())
#    test.svd = svd.transform(test.vecs.toarray())
#    train.svd.shape, test.svd.shape
    
#    nbrs_svd = NearestNeighbors(n_neighbors=num_neighs, algorithm='brute', metric=metric).fit(train.svd)
#    nbrs_svd_classifier = KNeighborsClassifier(num_neighs).fit(train.svd, train.target)
    
#    #print('#Train svd accuracy is ', accuracy_score(train.target, nbrs_svd_classifier.predict(train.svd))," @ ", x, "SVD", "(using TFIDF)")
#    print('#Test svd accuracy is ', accuracy_score(test.target, nbrs_svd_classifier.predict(test.svd)), " @ ", x, "SVD", "(using TFIDF)")    

#features = TfidfVectorizer(lowercase=True, stop_words = 'english', min_df=2, ngram_range = (1,2))
#Train svd accuracy is  0.6818926296633303  @  500 SVD (using TFIDF)
#Train svd accuracy is  0.6746132848043676  @  300 SVD (using TFIDF)
#Train svd accuracy is  0.6668789808917197  @  100 SVD (using TFIDF)
#Train svd accuracy is  0.6637852593266605  @  1 SVD (using TFIDF)

#Test svd accuracy is  0.4963609898107715  @  500 SVD (using TFIDF)
#Test svd accuracy is  0.5189228529839883  @  300 SVD (using TFIDF)
#Test svd accuracy is  0.5604075691411936  @  100 SVD (using TFIDF)
#Test svd accuracy is  0.5352983988355168  @  1 SVD (using TFIDF)

#-----------------------------------------------------------------------------------

#features = TfidfVectorizer(lowercase=True, stop_words = 'english', min_df=2, max_df = 0.5, ngram_range = (1,2))
#Train svd accuracy is  0.6651501364877161  @  500 SVD (using TFIDF)
#Train svd accuracy is  0.6665150136487716  @  300 SVD (using TFIDF)
#Train svd accuracy is  0.6695177434030937  @  100 SVD (using TFIDF)
#Train svd accuracy is  0.6608735213830755  @  1 SVD (using TFIDF)


#Test svd accuracy is  0.4916302765647744  @  500 SVD (using TFIDF)
#Test svd accuracy is  0.5207423580786026  @  300 SVD (using TFIDF)
#Test svd accuracy is  0.5655021834061136  @  100 SVD (using TFIDF)
#Test svd accuracy is  0.5582241630276564  @  1 SVD (using TFIDF)

#-----------------------------------------------------------------------------------
#features = TfidfVectorizer(lowercase=True, stop_words = 'english', min_df=2, max_df = 0.8, ngram_range = (1,2))
#Train svd accuracy is  0.6710646041856233  @  500 SVD (using TFIDF)
#Train svd accuracy is  0.6675159235668789  @  300 SVD (using TFIDF)
#Train svd accuracy is  0.6667879890809827  @  100 SVD (using TFIDF)
#Train svd accuracy is  0.6610555050045496  @  1 SVD (using TFIDF)

#Test svd accuracy is  0.49344978165938863  @  500 SVD (using TFIDF)
#Test svd accuracy is  0.5189228529839883  @  300 SVD (using TFIDF)
#Test svd accuracy is  0.5716885007278021  @  100 SVD (using TFIDF)
#Test svd accuracy is  0.5604075691411936  @  1 SVD (using TFIDF)

#-----------------------------------------------------------------------------------

#features = TfidfVectorizer(lowercase=True, stop_words = 'english', min_df=2, max_df = 0.9, ngram_range = (1,2))
#Train svd accuracy is  0.6769790718835305  @  500 SVD (using TFIDF)
#Train svd accuracy is  0.670336669699727  @  300 SVD (using TFIDF)
#Train svd accuracy is  0.6696087352138308  @  100 SVD (using TFIDF)
#Train svd accuracy is  0.6609645131938126  @  1 SVD (using TFIDF)

#Test svd accuracy is  0.48544395924308587  @  500 SVD (using TFIDF)
#Test svd accuracy is  0.5207423580786026  @  300 SVD (using TFIDF)
#Test svd accuracy is  0.5596797671033479  @  100 SVD (using TFIDF)
#Test svd accuracy is  0.5527656477438136  @  1 SVD (using TFIDF)

#-----------------------------------------------------------------------------------

#features = TfidfVectorizer(lowercase=True, stop_words = 'english', min_df=2, max_df = 0.7, ngram_range = (1,2))
#Train svd accuracy is  0.6691537761601456  @  500 SVD (using TFIDF)
#Train svd accuracy is  0.66196542311192  @  300 SVD (using TFIDF)
#Train svd accuracy is  0.6695177434030937  @  100 SVD (using TFIDF)
#Train svd accuracy is  0.664695177434031  @  1 SVD (using TFIDF)

#Test svd accuracy is  0.4923580786026201  @  500 SVD (using TFIDF)
#Test svd accuracy is  0.5207423580786026  @  300 SVD (using TFIDF)
#Test svd accuracy is  0.5585880640465793  @  100 SVD (using TFIDF)
#Test svd accuracy is  0.5567685589519651  @  1 SVD (using TFIDF)



#%% KNN SVD vecs (features gained using CountVectorizer)

num_neighs = 10
metric = 'cosine'

features = CountVectorizer(lowercase=True, stop_words='english', min_df = 2, max_df = 0.8, ngram_range = (1,2))
train.vecs = features.fit_transform(train.data)
test.vecs = features.transform(test.data)

#factors = TruncatedSVD(500).fit_transform(train.vecs.toarray())
#do_plot(factors)

train_acc = []
test_acc = []

svd_pass =  [300,200,100,1]

for x in svd_pass:
    
    svd = TruncatedSVD(x)
    train.svd = svd.fit_transform(train.vecs.toarray())
    test.svd = svd.transform(test.vecs.toarray())
    train.svd.shape, test.svd.shape
    
    nbrs_svd = NearestNeighbors(n_neighbors = num_neighs, algorithm='brute', metric = metric).fit(train.svd)
    nbrs_svd_classifier = KNeighborsClassifier(num_neighs).fit(train.svd, train.target)
    
    #print('#Train svd accuracy is ', accuracy_score(train.target, nbrs_svd_classifier.predict(train.svd)), " @ ", x, "SVD", "(using CountVectorizer)")
    #print('#Test svd accuracy is ', accuracy_score(test.target, nbrs_svd_classifier.predict(test.svd)), " @ ", x, "SVD", "(using CountVectorizer)")
    
    y = accuracy_score(train.target, nbrs_svd_classifier.predict(train.svd))
    train_acc.append(y)
    
    x = accuracy_score(test.target, nbrs_svd_classifier.predict(test.svd))
    test_acc.append(x)
    
counter = 0
for x in train_acc:
    print('#Train svd accuracy is ', x, " @ ", svd_pass[int(counter)], "SVD", "(using CountVectorizer)")
    counter += 1

counter = 0
for x in test_acc:
    print('#Test svd accuracy is ', x, " @ ", svd_pass[int(counter)], "SVD", "(using CountVectorizer)")
    counter += 1
    
    
    
#for x in svd_pass:
#    
#    svd = TruncatedSVD(x)
#    train.svd = svd.fit_transform(train.vecs.toarray())
#    test.svd = svd.transform(test.vecs.toarray())
#    train.svd.shape, test.svd.shape
#    
#    nbrs_svd = NearestNeighbors(n_neighbors=num_neighs, algorithm='brute', metric=metric).fit(train.svd)
#    nbrs_svd_classifier = KNeighborsClassifier(num_neighs).fit(train.svd, train.target)
#    
#    #print('#Train svd accuracy is ', accuracy_score(train.target, nbrs_svd_classifier.predict(train.svd)), " @ ", x, "SVD", "(using CountVectorizer)")
#    print('#Test svd accuracy is ', accuracy_score(test.target, nbrs_svd_classifier.predict(test.svd)), " @ ", x, "SVD", "(using CountVectorizer)")
    
#features = CountVectorizer(lowercase=True, stop_words='english', min_df=2, ngram_range = (1,2))    
#Train svd accuracy is  0.6509554140127388  @  500 SVD (using CountVectorizer)
#Train svd accuracy is  0.6564149226569609  @  300 SVD (using CountVectorizer)
#Train svd accuracy is  0.6667879890809827  @  100 SVD (using CountVectorizer)
#Train svd accuracy is  0.6551410373066424  @  1 SVD (using CountVectorizer)

#Test svd accuracy is  0.5258369723435226  @  500 SVD (using CountVectorizer)
#Test svd accuracy is  0.5323871906841339  @  300 SVD (using CountVectorizer)
#Test svd accuracy is  0.5418486171761281  @  100 SVD (using CountVectorizer)
#Test svd accuracy is  0.5534934497816594  @  1 SVD (using CountVectorizer)

#-------------------------------------------------------------------------------------------------

#features = CountVectorizer(lowercase=True, stop_words='english', min_df=2, max_df = 0.5, ngram_range = (1,2))
#Train svd accuracy is  0.6478616924476797  @  500 SVD (using CountVectorizer)
#Train svd accuracy is  0.6564149226569609  @  300 SVD (using CountVectorizer)
#Train svd accuracy is  0.665059144676979   @  100 SVD (using CountVectorizer)
#Train svd accuracy is  0.6551410373066424  @  1 SVD (using CountVectorizer)

#Test svd accuracy is  0.517467248908297  @  500 SVD (using CountVectorizer)
#Test svd accuracy is  0.5356622998544396  @  300 SVD (using CountVectorizer)
#Test svd accuracy is  0.5462154294032023  @  100 SVD (using CountVectorizer)
#Test svd accuracy is  0.5564046579330422  @  1 SVD (using CountVectorizer)

#-------------------------------------------------------------------------------------------------

#features = CountVectorizer(lowercase=True, stop_words='english', min_df = 2, max_df = 0.8, ngram_range = (1,2))
#Train svd accuracy is  0.6524112829845314  @  500 SVD (using CountVectorizer)
#Train svd accuracy is  0.6558689717925387  @  300 SVD (using CountVectorizer)
#Train svd accuracy is  0.6678798908098271  @  100 SVD (using CountVectorizer)
#Train svd accuracy is  0.6550500454959054  @  1 SVD (using CountVectorizer)

#Test svd accuracy is  0.5254730713245997  @  500 SVD (using CountVectorizer)
#Test svd accuracy is  0.5436681222707423  @  300 SVD (using CountVectorizer)
#Test svd accuracy is  0.5483988355167394  @  100 SVD (using CountVectorizer)
#Test svd accuracy is  0.5545851528384279  @  1 SVD (using CountVectorizer)


#Train svd accuracy is  0.6805277525022748  @  500 SVD (using TFIDF)
#Train svd accuracy is  0.6670609645131939  @  300 SVD (using TFIDF)
#Train svd accuracy is  0.6651501364877161  @  100 SVD (using TFIDF)
#Train svd accuracy is  0.6498635122838945  @  1 SVD (using TFIDF)

#Test svd accuracy is  0.4890829694323144  @  500 SVD (using TFIDF)
#Test svd accuracy is  0.5109170305676856  @  300 SVD (using TFIDF)
#Test svd accuracy is  0.5574963609898108  @  100 SVD (using TFIDF)
#Test svd accuracy is  0.5403930131004366  @  1 SVD (using TFIDF)

#-----------------------------------------------------------------------------------

#features = TfidfVectorizer(lowercase=True, stop_words = 'english', max_df = 0.8, ngram_range = (1,2))
#Train svd accuracy is  0.6458598726114649  @  500 SVD (using CountVectorizer)
#Train svd accuracy is  0.654231119199272  @  300 SVD (using CountVectorizer)
#Train svd accuracy is  0.6643312101910828  @  100 SVD (using CountVectorizer)
#Train svd accuracy is  0.6550500454959054  @  1 SVD (using CountVectorizer)

#Test svd accuracy is  0.5302037845705968  @  500 SVD (using CountVectorizer)
#Test svd accuracy is  0.5291120815138283  @  300 SVD (using CountVectorizer)
#Test svd accuracy is  0.5498544395924309  @  100 SVD (using CountVectorizer)
#Test svd accuracy is  0.5531295487627366  @  1 SVD (using CountVectorizer)

#-----------------------------------------------------------------------------------

#features = TfidfVectorizer(lowercase=True, stop_words = 'english', min_df=2, max_df = 0.8, ngram_range = (1,2), max_features = 400)
#Train svd accuracy is  0.6489535941765241  @  500 SVD (using CountVectorizer)
#Train svd accuracy is  0.6555959963603276  @  300 SVD (using CountVectorizer)
#Train svd accuracy is  0.6656050955414012  @  100 SVD (using CountVectorizer)
#Train svd accuracy is  0.6551410373066424  @  1 SVD (using CountVectorizer)

#Test svd accuracy is  0.5302037845705968  @  500 SVD (using CountVectorizer)
#Test svd accuracy is  0.5400291120815138  @  300 SVD (using CountVectorizer)
#Test svd accuracy is  0.5534934497816594  @  100 SVD (using CountVectorizer)
#Test svd accuracy is  0.5549490538573508  @  1 SVD (using CountVectorizer)

#-----------------------------------------------------------------------------------

#features = TfidfVectorizer(lowercase=True, stop_words = 'english', min_df=2, max_df = 0.8, ngram_range = (1,2), max_features = 1000)
#Train svd accuracy is  0.6526842584167425  @  500 SVD (using CountVectorizer)
#Train svd accuracy is  0.6565059144676979  @  300 SVD (using CountVectorizer)
#Train svd accuracy is  0.6686078252957234  @  100 SVD (using CountVectorizer)
#Train svd accuracy is  0.6547770700636942  @  1 SVD (using CountVectorizer)

#Test svd accuracy is  0.5323871906841339  @  500 SVD (using CountVectorizer)
#Test svd accuracy is  0.5331149927219796  @  300 SVD (using CountVectorizer)
#Test svd accuracy is  0.5422125181950509  @  100 SVD (using CountVectorizer)
#Test svd accuracy is  0.5545851528384279  @  1 SVD (using CountVectorizer)

#-----------------------------------------------------------------------------------

#features = TfidfVectorizer(lowercase=True, stop_words = 'english', min_df=2, max_df = 0.8, ngram_range = (1,2), max_features = 100) 
#Train svd accuracy is  0.6499545040946315  @  500 SVD (using CountVectorizer)
#Train svd accuracy is  0.6585987261146496  @  300 SVD (using CountVectorizer)
#Train svd accuracy is  0.6681528662420382  @  100 SVD (using CountVectorizer)
#Train svd accuracy is  0.6553230209281165  @  1 SVD (using CountVectorizer)

#Test svd accuracy is  0.5283842794759825  @  500 SVD (using CountVectorizer)
#Test svd accuracy is  0.534570596797671  @  300 SVD (using CountVectorizer)
#Test svd accuracy is  0.5327510917030568  @  100 SVD (using CountVectorizer)
#Test svd accuracy is  0.5549490538573508  @  1 SVD (using CountVectorizer)

#-----------------------------------------------------------------------------------

#features = TfidfVectorizer(lowercase=True, stop_words = 'english', min_df=2, max_df = 0.8, ngram_range = (1,2), max_features = 25) 
#Train svd accuracy is  0.6505914467697907  @  500 SVD (using CountVectorizer)
#Train svd accuracy is  0.6612374886260236  @  300 SVD (using CountVectorizer)
#Train svd accuracy is  0.6642402183803457  @  100 SVD (using CountVectorizer)
#Train svd accuracy is  0.6549590536851684  @  1 SVD (using CountVectorizer)

#Test svd accuracy is  0.517467248908297  @  500 SVD (using CountVectorizer)
#Test svd accuracy is  0.5367540029112081  @  300 SVD (using CountVectorizer)
#Test svd accuracy is  0.5480349344978166  @  100 SVD (using CountVectorizer)
#Test svd accuracy is  0.5549490538573508  @  1 SVD (using CountVectorizer)




 #%%  another run of W2V KNN (this one was better)

train.toks = []
for s in train.data:
  train.toks.append(gensim.utils.simple_preprocess(s))
test.toks = []
for s in test.data:
  test.toks.append(gensim.utils.simple_preprocess(s))
  
  from gensim.models.word2vec import Word2Vec
model = Word2Vec(train.toks, vector_size=100, window=5, min_count=5, workers=4,epochs = 1000)
word_vectors = model.wv
del model
#%% KNN w2V (this was better)
#word_vectors = all_model.wv
train.w2v = np.zeros((len(train.data), word_vectors['good'].shape[0]))
idx = 0

for s in train.toks:
  ws = []
  for w in s:
    if w in word_vectors:
      ws.append(w)
  if len(ws) is not 0:
    train.w2v[idx] = np.mean(word_vectors[ws], axis=0)
  idx += 1
  
test.w2v = np.zeros((len(test.data), word_vectors['good'].shape[0]))
idx = 0

for s in test.toks:
  ws = []
  for w in s:
    if w in word_vectors:
      ws.append(w)
  if len(ws) is not 0:
    test.w2v[idx] = np.mean(word_vectors[ws], axis=0)
  idx += 1
  
  
nbrs_w2v = NearestNeighbors(n_neighbors = num_neighs, algorithm='brute', metric=metric).fit(train.w2v)
nbrs_w2v_classifier = KNeighborsClassifier(num_neighs).fit(train.w2v, train.target)

train_score = accuracy_score(train.target, nbrs_w2v_classifier.predict(train.w2v))
#0.6733393994540491

test_score = accuracy_score(test.target, nbrs_w2v_classifier.predict(test.w2v))
#0.5633187772925764

print('W2V Train Accuracy is ', accuracy_score(train.target, nbrs_w2v_classifier.predict(train.w2v)))
print('W2V Test Accuracy is ', accuracy_score(test.target, nbrs_w2v_classifier.predict(test.w2v)))

#W2V Train Accuracy = 0.6733393994540491
#W2V Test Accuracy is  0.5618631732168851
print(train_score, test_score)


#1000 epochs model I built
#W2V Train Accuracy is  0.6747952684258417
#W2V Test Accuracy is  0.5738719068413392


#profs model at 1000 epochs
#W2V Train Accuracy is  0.6817106460418563
#W2V Test Accuracy is  0.5644104803493449


#%% logistic regression  (features created with TFIDF)
num_neighs = 10
metric = 'cosine'

features = TfidfVectorizer(lowercase=True, stop_words='english', min_df=1, max_df=0.8, ngram_range = (1,2))
train.vecs = features.fit_transform(train.data)
test.vecs = features.transform(test.data)

#feats.steps[0][1].get_feature_names()

#print(train.vecs[1])

lr_model = LogisticRegression(C= 0.1, solver = 'newton-cg', penalty = 'l2')
lr_model.fit(train.vecs,train.target)
#lr_model = make_pipeline(TfidfVectorizer(), LogisticRegression())
#lr_model.fit(train.data, train.target)

train_preds = lr_model.predict(train.vecs)
train_f1 = f1_score(train.target, train_preds, average='micro')
train_f1
#train F1 score = 0.7364245159411851


print('LR Train accuracy is ', accuracy_score(train.target, lr_model.predict(train.vecs)))
#LR Train accuracy is  0.7770700636942676

test_preds = lr_model.predict(test.vecs)
test_f1 = f1_score(test.target, test_preds, average='micro')
test_f1
# test F1 score = 0.6098981077147017


print('LR Test accuracy is ', accuracy_score(test.target, lr_model.predict(test.vecs)))
#LR Test accuracy score =  0.6000727802037845



print(train_f1, test_f1)

#%% TFIDF regression with ELI5 


new_train = train.reset_index()
new_test = test

new_train = new_train.reset_index(drop = True)
new_test = new_test.reset_index(drop = True)

feats = make_pipeline(CustomFeats(), TfidfVectorizer())

feats.steps[0][1].get_feature_names()


lr_model = LogisticRegression(C=0.1)
vec = CountVectorizer()
pipe = make_pipeline(vec, lr_model)
pipe.fit(new_train.data, new_train.target)
train_preds = pipe.predict(new_train.data)

train_f1 = f1_score(new_train.target, train_preds, average='micro')
test_preds = pipe.predict(new_test.data)
test_f1 = f1_score(new_test.target, test_preds, average='micro')
train_f1, test_f1  
print('LR Train accuracy is ', accuracy_score(new_train.target, train_preds))
#LR Train accuracy is  0.7840764331210192

print('LR Test accuracy is ', accuracy_score(new_test.target,test_preds))
#LR Test accuracy is  0.5884279475982532

eli5.show_weights(pipe, top=10, target_names = new_test.target)

idx = 10
x = new_test.data[idx]
#print(test.data[idx])
print(new_test.target[new_test.target[idx]])
eli5.show_prediction(lr_model, new_test.data[idx], vec=vec, target_names = new_test.target)


#%% Logistic regression (features created with Bag of Words/Count vectorizer)

num_neighs = 10
metric = 'cosine'

features = CountVectorizer(lowercase=True, stop_words='english', min_df=2, ngram_range = (1,2))
train.vecs = features.fit_transform(train.data)
test.vecs = features.transform(test.data)

#feats.steps[0][1].get_feature_names()

#print(train.vecs[1])

lr_model = LogisticRegression(C=1)
lr_model.fit(train.vecs,train.target)
#lr_model = make_pipeline(CountVectorizer(), LogisticRegression())
#lr_model.fit(train.data, train.target)

train_preds = lr_model.predict(train.vecs)
train_f1 = f1_score(train.target, train_preds, average='micro')
train_f1
#train F1 score = 0.9322111010009099
print('LR Train accuracy is ', accuracy_score(train.target, lr_model.predict(train.vecs)))
#0.9322111010009099

test_preds = lr_model.predict(test.vecs)
test_f1 = f1_score(test.target, test_preds, average='micro')
test_f1
# test F1 score = 0.5695050946142649

print('LR Test accuracy is ', accuracy_score(test.target, lr_model.predict(test.vecs)))
#LR Test accuracy is  0.5695050946142649

print(train_f1, test_f1)
#0.9322111010009099 0.5695050946142649

#%% Results

####### Logistic regression (features created with Bag of Words/Count vectorizer) #############
#train_f1, test_f1 = 0.9322111010009099 0.5695050946142649

####### Logistic regression  (features created with TFIDF) ##########
#train_f1, test_f1 = 0.7770700636942676 0.6000727802037845

####### W2V (original model) KNN ##########
#train_score, test_score = 0.6733393994540491, 0.5618631732168851

###### KNN Vecs Classification (features created with CountVectorizer) #########
#Train, test accuracies = 0.6626933575978162, 0.5549490538573508

###### KNN vecs classification (features created with TFIDF) #########
#train, test accuracies = 0.6847133757961783, 0.5862445414847162

###### KNN SVD vecs (features gained using CountVectorizer) ##########
#train, test accuracies
#svd accuracy is 0.5214701601164483 @ 500 SVD
#svd accuracy is  0.5389374090247453 @ 300 svd
#svd accuracy is  0.5524017467248908 @ 100 svd
#svd accuracy is  0.5542212518195051 @ 1 svd


###### KNN SVD vecs (features gained using TFIDF) #########
#train, test accuracies
#svd accuracy is 0.4799854439592431 @ 500 SVD
#svd accuracy is  0.5229257641921398 @ 300 svd
#svd accuracy is  0.5534934497816594 @ 100 svd
#svd accuracy is  0.5567685589519651 @ 1 svd