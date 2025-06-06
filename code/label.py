import random
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from gensim.models.doc2vec import Doc2Vec


def print_counts(array):
    unique, counts = np.unique(array, axis=0, return_counts=True)
    for count in list(zip(unique, counts)):
        print(count)


def main():
    '''
    Labels the given dataset and returns only interesting fields
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

    # load data
    train_data = pd.read_csv('data/label_binary.csv')
    true_labels = (train_data.dropna(how='any')[labels].to_numpy())
    text_vectors = np.load('data/docvecs_50.npy')
    text_vectors = text_vectors[:true_labels.shape[0]]
    train_vecs, test_vecs, train_labels, test_labels = train_test_split(
        text_vectors, true_labels, test_size=0.3, shuffle=False
    )

    # load doc2vec model and vectorise
    doc2vec = Doc2Vec.load('models/doc2vec_50.model')
    vectors = data['body'].apply(
        lambda x: doc2vec.infer_vector(x.split())
    )
    vectors = np.vstack(vectors)
    vectors = vectors + abs(vectors.min())

    # load best model
    model = KNeighborsClassifier(
        n_neighbors=5, weights='uniform', algorithm='kd_tree', leaf_size=20,
        p=1.5
    )

    # fit and predict labels
    model.fit(train_vecs, train_labels)
    predicted_labels = model.predict(vectors)
    print_counts(predicted_labels)
    df = pd.concat([
        data,
        pd.DataFrame(predicted_labels, columns=labels)
    ], axis=1)

    df.to_csv('data/threads1000_format_labeled.csv', index=False)


if __name__ == '__main__':
    main()
