import pickle
from centroid import *
from weight import  *
import numpy as np
import pickle

def cosineSimilarity(d1,du):
    product = dotProduct(d1, du)
    norm = dotProduct(d1, d1) **0.5
    similarity = product/norm
    return similarity

def topScore(d1, centroid, n):
    simil = {}
    for k,v in centroid.items():
        simil[k] = cosineSimilarity(d1, v)
    sortSimil = sorted([(value,key) for (key,value) in simil.items()], reverse= True)
    result = sortSimil[:n]
    return result

if __name__ == '__main__':
    h = 1.5
    b = 0.75
    print 'loading centroid'
    centroid = loadPickle('./Data/centroid.pkl')
    length = loadPickle('./Data/length.pkl')
    aveL = np.mean(length.values())

    infile = "/Users/mchen/Rocchio/Data/topic_content.txt"
    fo = open(infile)
    label = []
    predict = []

    for line in fo.readlines():
        topic, _, content = line.strip().split('\t')
        l, tfDict = tf(content)
        tfwDict = termFrequencyWeight(tfDict, h, b, l, aveL)
        topN = topScore(tfwDict, centroid, 10)
        label.append(topic)
        predict.append(topN)
    fo.close()
    pkl1 = open('./Result/trueLabel.pkl', 'wb')
    pickle.dump(label, pkl1)
    pkl1.close()

    pkl2 = open('./Result/predict.pkl', 'wb')
    pickle.dump(predict, pkl2)
    pkl2.close()
