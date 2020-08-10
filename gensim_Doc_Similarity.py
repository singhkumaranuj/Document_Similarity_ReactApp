from collections import defaultdict
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import re
import json
import docx2txt
import pickle
import spacy
from flask import Flask,jsonify,request
from flask_cors import CORS,cross_origin
from gensim import models,corpora,similarities
from gensim.utils import simple_preprocess
from sklearn.metrics.pairwise import cosine_similarity
nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)
CORS(app)

#document details
#mentioned your path here.
doc_1 = r'.docx'
doc_2 = r'.docx'
doc_3 = r'.docx'

documents_List=['doc_1','doc_2','doc_3']

#preprocessing functions
def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase
def preprocessing(text):
    from tqdm import tqdm
    import re
    from bs4 import BeautifulSoup
    stopwords = set(
        ['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll",
         "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her',
         'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what',
         'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be',
         'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but',
         'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
         'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
         'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
         'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same',
         'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now',
         'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn',
         "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't",
         'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren',
         "weren't", 'won', "won't", 'wouldn', "wouldn't"])

    preprocessed_data = []
    sentance = text
    sentance = re.sub(r"http\S+", "", sentance)
    sentance = BeautifulSoup(sentance, 'lxml').get_text()
    sentance = decontracted(sentance)
    sentance = re.sub("\S*\d\S*", "", sentance).strip()
    sentance = re.sub('[^A-Za-z]+', ' ', sentance)
    # https://gist.github.com/sebleier/554280
    sentance = ' '.join(e.lower() for e in sentance.split() if e.lower() not in stopwords)
    preprocessed_data.append(sentance.strip())
    return ' '.join(preprocessed_data)


#document preprocessing
final_preprocessed_docs =[]
doc_1 = docx2txt.process(doc_1)
preprocessed_doc_1=preprocessing(doc_1)
final_preprocessed_docs.append(preprocessed_doc_1)

doc_2 = docx2txt.process(doc_2)
preprocessed_doc_2=preprocessing(doc_2)
final_preprocessed_docs.append(preprocessed_doc_2)

doc_3 = docx2txt.process(doc_3)
preprocessed_doc_3=preprocessing(doc_3)
final_preprocessed_docs.append(preprocessed_doc_3)



preprocessed_texts = [simple_preprocess(text) for text in final_preprocessed_docs]
dictionary = corpora.Dictionary(preprocessed_texts) # assigning id to each and every token
bow_corpus = [dictionary.doc2bow(val) for val in preprocessed_texts] # bow corpus for each tokenized word i.e. freq of word in corpus ..means tokenid and their respective frequency
lsi = models.LsiModel(bow_corpus, id2word=dictionary)
word_counts = [[(dictionary[id], count) for id, count in line] for line in bow_corpus]

@app.route('/findLSISimilarity/',methods=['GET', 'POST'])
@cross_origin(origin='*')
def find_similarity():
    final_dict = {}
    final_q_processed=[]
    new_q_processed = set()
    a = request.get_json(force=True)
    q = a["query"]
    q_processed = list([preprocessing(q)])
    #spacy start
    # spacy find noun  in question
    doc = nlp(' '.join(q_processed))
    new_q_processed = set()
    for w in list(doc.sents)[0]:
        if w.pos_ == 'NOUN' or w.pos_=='PROPN':
            new_q_processed.add(str(w))
        else:
            new_q_processed.add(' '.join(q_processed))
    final_q_processed = list(new_q_processed)
    print('---0----',final_q_processed)
    # doc = ' '.join(final_q_processed)
    doc = []
    for val in final_q_processed:
        for v in val.split():
            if v not in doc:
                doc.append(v)
    doc =' '.join(doc)
    print('----1-----',doc)
    # spacy end
    vec_bow = dictionary.doc2bow(doc.lower().split())
    #vec_bow = [dictionary.doc2bow(val) for val in doc]
    vec_lsi = lsi[vec_bow]
    index = similarities.MatrixSimilarity(lsi[bow_corpus])
    sims = index[vec_lsi]  # perform a similarity query against the corpus
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    for i, s in enumerate(sims):
        final_dict[documents_List[s[0]]] = float(s[1])
    intent_dict = {}
    intent = list(final_dict.keys())[0]
    intent_dict['doc_name'] = intent
    intent_dict['score'] = final_dict[intent]
    return jsonify(intent_dict)

if __name__=="__main__":
   app.run()
    #app.run(host='0.0.0.0',port=5000)