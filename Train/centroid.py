import pickle

def loadPickle(infile):
    pkl_file = open(infile, 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()
    return data

def summation(d1, d2):
    if len(d1)> len(d2):
        return summation(d2,d1)
    else:
        d = d2
        for k,v in d1.items():
            if d.has_key(k):
                d[k] += v
            else:
                d[k] = v
        return d
def dotProduct(d1, d2):
    if len(d1) < len(d2):
        return dotProduct(d2, d1)
    else:
        return sum(d1.get(f, 0) * v for f, v in d2.items())

def product(d1,d2):
    if len(d1) < len(d2):
        return product(d2, d1)
    else:
        d = {}
        for k,v in d2.items():
            if d1.has_key(k):
                d[k] = d1[k]*d2[k]
        return d
def scale(d, s):
    d1 = {}
    for f, v in d.items():
        d1[f] = v*s
    return d1

def categoryCentroid(tf, idf, label):
    centroid = {}
    size = {}
    for k in label.keys():
        tfidf = product(tf[k], idf)
        if centroid.has_key(label[k]):
            centroid[label[k]] = summation(centroid[label[k]], tfidf)
            size[label[k]] +=1
        else:
            centroid[label[k]] = {}
            centroid[label[k]] = summation(centroid[label[k]], tfidf)
            size[label[k]] = 1
    mu = {}
    for key,v in centroid.items():
        s = size[key]
        aveVector = scale(v, 1/float(s))
        norm = dotProduct(aveVector, aveVector) ** 0.5
        unit = scale(aveVector, 1/norm)
        mu[key] = unit
    return mu
            
if __name__ == "__main__":
    print 'loading label'
    label = loadPickle('./Data/label_test.pkl')
    print 'loading tfw'
    tf = loadPickle('./Data/tfw_test.pkl')
    print 'loading idf'
    idf = loadPickle('./Data/idf_test.pkl')
    print 'calculating centroid'
    centroid = categoryCentroid(tf, idf, label)

    output = open('./Data/centroid_test.pkl', 'wb')
    pickle.dump(centroid, output)
    output.close()
