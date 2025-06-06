import random
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from gensim.models.doc2vec import Doc2Vec


def print_counts(array):
    unique, counts = np.unique(array, axis=0, return_counts=True)
    for count in list(zip(unique, counts)):
        print(count)


def main():
    '''
    Code for troubleshooting when certain models only classified singular
     label combinations like 1,1,1,1
    '''
    # set seeds
    np.random.seed(1)
    random.seed(1)

    data = pd.read_csv('data/threads1000_format.csv')
    data = data[[
        'author', 'body', 'created_utc', 'deltas', 'discussion_number',
        'downs', 'gilded', 'id', 'level', 'name', 'score', 'title', 'ups'
    ]]

    data['body'] = data['body'].astype(str)

    labels = ['agency', 'event_sequencing', 'world_making', 'story']

    train_data = pd.read_csv('data/label_binary.csv')
    true_labels = (train_data.dropna(how='any')[labels].to_numpy())
    # print(np.unique(true_labels, axis=0, return_counts=True))
    text_vectors = np.load('data/docvecs_50.npy')
    text_vectors = text_vectors[:true_labels.shape[0]]
    train_vecs, test_vecs, train_labels, test_labels = train_test_split(
        text_vectors, true_labels, test_size=0.3, shuffle=False
    )
    print_counts(test_labels)

    doc2vec = Doc2Vec.load('models/doc2vec_50.model')
    vectors = data['body'].apply(
        lambda x: doc2vec.infer_vector(x.split())
    )
    vectors = np.vstack(vectors)
    vectors = vectors + abs(vectors.min())
    print(vectors)
    print(len(np.unique(vectors, axis=0)))

    models = {
        'svc': lambda params: MultiOutputClassifier(SVC(
            C=params[0], kernel=params[1], degree=params[2], gamma=params[3],
            coef0=params[4], random_state=1
        )),
        'knn': lambda params: KNeighborsClassifier(
            n_neighbors=params[0], weights=params[1], algorithm=params[2],
            leaf_size=params[3], p=params[4]
        ),
        'random forest': lambda params: RandomForestClassifier(
            n_estimators=params[0], criterion=params[1],
            min_samples_split=params[2], min_samples_leaf=params[3],
            max_features=params[4], min_impurity_decrease=params[5],
            ccp_alpha=params[6], random_state=1
        ),
        'multinomialnb': lambda params: MultiOutputClassifier(MultinomialNB(
            alpha=params[0], force_alpha=params[1]
        )),
        'guassiannb': lambda params: MultiOutputClassifier(GaussianNB(
            var_smoothing=params[0]
        ))
    }

    parameters = {
        'guassiannb': (1e-10,),
        'knn': (5, 'uniform', 'kd_tree', 20, 1.5),
        'multinomialnb': (0.5, False),
        'random forest': (150, 'gini', 2, 4, None, 0, 0.1),
        'svc': (1.25, 'poly', 5, 'auto', 0.0),
    }

    # model = MultiOutputClassifier(SVC(
    #     C=0.75, kernel='poly', degree=5, gamma='auto', coef0=0.0,
    #     random_state=1
    # ))

    for model_type, parameter in parameters.items():
        model = models[model_type](parameter)
        model.fit(train_vecs, train_labels)
        # print(np.unique(model.predict(test_vecs), axis=0))
        # print_counts(model.predict(test_vecs))

        predicted_labels = model.predict(vectors)
        # print(predicted_labels)
        print(model_type, parameter)
        # print(np.unique(predicted_labels, axis=0))
        print_counts(predicted_labels)


if __name__ == '__main__':
    main()
