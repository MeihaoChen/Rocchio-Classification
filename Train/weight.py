import pickle
import string
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
from nltk.corpus import stopwords
from collections import Counter

def parseDocument(d):
    stop = stopwords.words('english')
    document = ''.join([i for i in d if i not in string.punctuation]).lower()
    token = [word for word in document.split(' ') if word not in stop]
    return token

def lancasterStemming(tokens):
    stemmed = []
    st = LancasterStemmer()
    for i in tokens:
        stemmed.append(st.stem(i))
    return stemmed
def tf(d):
    token = lancasterStemming(parseDocument(d))
    length = len(token)
    term_frequency = Counter(token)
    return length, term_frequency

def termFrequencyWeight(tf, h, b, l, avel):
    tfWeight = {}
    for key,value in tf.items():
        tfw = (value * (h+1))/(value + h*(1-b+b*l/avel))
        tfWeight[key] = tfw
    return tfWeight

def idf(alldata):
    N = len(alldata)
    invertedIndex = {}
    df = {}
    for i in alldata.values():
        words = i.keys()
        for j in words:
            if invertedIndex.has_key(j):
                invertedIndex[j].append(i)
                df[j] += 1
            else:
                invertedIndex[j] = []
                invertedIndex[j].append(i)
                df[j] = 1
    idf = {}
    for k,v in df.items():
        logIDF = np.log((N-v+0.5)/(v+0.5))
        idf[k] = logIDF
    return idf

if __name__ == "__main__":
    b = 0.75
    q = 1.5

    print "opening file"
    fo = open('../Data/topic_content.txt')
    label = {}
    termFrequency = {}
    length = {}
    print "generating length, tf, and label dictionary"
    for line in fo.readlines():
        topic_id, ID, text = line.strip().split('\t')
        label[ID] = topic_id
        l, termfreq = tf(text)
        length[ID] = l
        termFrequency[ID] = termfreq
    fo.close()
    averageL = np.mean(length.values())
    
    tfw = {}
    print "calculating weighted term frequency"
    for k in length.keys():
        documentTermFrequency = termFrequency[k]
        documentLength = length[k]
        tfw[k] = termFrequencyWeight(documentTermFrequency, q, b, documentLength, averageL)
    print "calculating inverseDocumentFrequency" 
    inverseDocumentFrequency = idf(termFrequency)

    print "writing to file"
    output = open('./Data/label.pkl', 'wb')
    pickle.dump(label, output)
    output.close()

    output = open('./Data/length.pkl', 'wb')
    pickle.dump(length, output)
    output.close()

    output = open('./Data/termFrequency.pkl', 'wb')
    pickle.dump(termFrequency, output)
    output.close()

    output = open('./Data/tfw.pkl', 'wb')
    pickle.dump(tfw, output)
    output.close()

    output = open('./Data/idf.pkl', 'wb')
    pickle.dump(inverseDocumentFrequency, output)
    output.close()
    print "done"

