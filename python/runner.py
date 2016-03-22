import numpy as np
import random
from sklearn import cross_validation
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB

def load_data(filename):
    d = []
    with open(filename) as ifh:
        for line in ifh:
            d.append([float(a) for a in line.strip().split(',')])
    return np.array(d)

def small_test(model, features, labels, k):
    trainsize = int(0.1 * len(features))
    results = []
    for _ in range(k):
        ordering = [a for a in range(len(features))]
        random.shuffle(ordering)
        model.fit(features[ordering[:trainsize]], labels[ordering[:trainsize]])
        predictions = model.predict(features[ordering[trainsize:]])
        results.append(
            sum(predictions == labels[ordering[trainsize:]])/len(predictions))
    return np.array(results)

def run():
    data = {
        'forests_d': load_data('../data/forests_d.data'),
        'vert': load_data('../data/vert.data')}
    models = {
        'linear_svm': svm.SVC(kernel='linear', C=1),
        'rbf_svm': svm.SVC(kernel='rbf', C=1)}
    for d in data:
        features = data[d][:, :-1]
        labels = data[d][:, -1]
        for m in models:
            scores = cross_validation.cross_val_score(
                models[m], features, labels, cv=4)
            '''
            scores = small_test(models[m], features, labels, 10)
            '''
            with open('../results/'+d+'.cross4.'+m, 'w') as ofh:
                ofh.write(' '.join([str(a) for a in scores]))
                ofh.write('\n')
                ofh.write(str(scores.mean()))

if __name__ == '__main__':
    run()

