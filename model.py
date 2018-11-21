import numpy as np


class NaiveBayesClassifier:

    def __init__(self):
        self.n_c_list = [0]
        self.n_jc = 0

    def fit(self, features, labels):
        '''
        features: (n_of_data, n_of_features)
        labels: (n_of_data, n_of_features)
        '''
        self.n_of_classes = np.unique(labels).shape[0]
        self.n_of_features = features.shape[1]

        n_of_data = features.shape[0]
        self.n_cs = labels.sum(axis=0)
        self.n_jcs = np.dot(features.T, labels)

        self.theata_jcs = self.n_jcs / n_of_data
        self.pi_cs = self.n_cs / n_of_data

    def predict(self, features):
        likelihoods = np.log1p(np.identity(2)[features], self.theata_jcs) + np.log(self.pi_cs)
        prob = np.exp(likelihoods - np.log1p(np.sum(np.exp(likelihoods))))
        return np.argmax(prob)

    def sorted_important_features_index(self):
        theata_j = np.tile(np.dot(self.theata_jcs, self.pi_cs), (self.n_of_classes, 1)).T
        matual_informations = self.theata_jcs * self.pi_cs \
            * np.log1p(np.divide(self.theata_jcs, theata_j)) \
            + (1 - self.theata_jcs) * self.pi_cs * np.log1p(np.divide(1 - self.theata_jcs, 1 - theata_j))

        return np.argsort(matual_informations.sum(1))
