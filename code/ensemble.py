import json
import random
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB


def main():
    '''
    Performs 2 ensemble methods for the classification task
    '''
    # set seeds
    np.random.seed(1)
    random.seed(1)

    # parameters of best performing parameter combinations from parameters.py
    parameters = {
        'medians': {
            200: [
                ('svc', (1.5, 'sigmoid', 3, 0.35, 0.5)),
                ('knn', (7, 'uniform', 'kd_tree', 40, 2.5)),
                ('random forest', (250, 'log_loss', 5, 5, None, 0.5, 0.1)),
                ('multinomialnb', (1.5, True)),
                ('guassiannb', (1e-11,)),
            ],
            100: [
                ('svc', (1.5, 'sigmoid', 3, 0.35, 0.5)),
                ('knn', (7, 'uniform', 'kd_tree', 40, 2.5)),
                ('random forest', (250, 'log_loss', 5, 5, None, 0.5, 0.1)),
                ('multinomialnb', (1.5, True)),
                ('guassiannb', (1e-11,)),
            ],
            50: [
                ('svc', (1.5, 'sigmoid', 3, 0.35, 0.5)),
                ('knn', (7, 'uniform', 'kd_tree', 40, 2.5)),
                ('random forest', (250, 'log_loss', 5, 5, None, 0.5, 0.1)),
                ('multinomialnb', (1.5, True)),
                ('guassiannb', (1e-11,)),
            ],
        },
        'binary': {
            200: [
                ('svc', (1.5, 'sigmoid', 3, 0.35, 0.5)),
                ('knn', (7, 'uniform', 'kd_tree', 40, 2.5)),
                ('random forest', (250, 'log_loss', 5, 5, None, 0.5, 0.1)),
                ('multinomialnb', (1.5, True)),
                ('guassiannb', (1e-11,)),
            ],
            100: [
                ('svc', (1.5, 'sigmoid', 3, 0.35, 0.5)),
                ('knn', (7, 'uniform', 'kd_tree', 40, 2.5)),
                ('random forest', (250, 'log_loss', 5, 5, None, 0.5, 0.1)),
                ('multinomialnb', (1.5, True)),
                ('guassiannb', (1e-11,)),
            ],
            50: [
                ('svc', (1.5, 'sigmoid', 3, 0.35, 0.5)),
                ('knn', (7, 'uniform', 'kd_tree', 40, 2.5)),
                ('random forest', (250, 'log_loss', 5, 5, None, 0.5, 0.1)),
                ('multinomialnb', (1.5, True)),
                ('guassiannb', (1e-11,)),
            ],
        },
    }

    # initialise labels
    labels = ['agency', 'event_sequencing', 'world_making', 'story']

    # setup model loading
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

    # initialise ensemble types
    ensemble_types = ['average', 'weighted_average', 'max_vote']

    # setup results dict
    results = {
        conversion: {
            vector_size: {
                ensemble_type: dict()
                for ensemble_type in ensemble_types
            }
            for vector_size in parameters[conversion]
        }
        for conversion in parameters
    }

    for conversion in parameters:
        # load data
        data = pd.read_csv(f'data/label_{conversion}.csv')

        # initialise unique classes for metrics
        label_classes = {
            label: data[label].unique() for label in labels
        }

        # select only needed labels
        true_labels = (data.dropna(how='any')[
            ['agency', 'event_sequencing', 'world_making', 'story']
        ].to_numpy())

        for vector_size in parameters[conversion]:
            print(f'Running on vector size {vector_size} and {conversion}')

            # load text vectors
            text_vectors = np.load(f'data/docvecs_{vector_size}.npy')
            text_vectors = text_vectors[:true_labels.shape[0]]

            # split dataset
            train_vecs, test_vecs, train_labels, test_labels = (
                train_test_split(
                    text_vectors, true_labels, test_size=0.3, shuffle=False
                )
            )

            # perform predictions for each model
            predicitions = []
            for model_type, parameter in parameters[conversion][vector_size]:
                model = models[model_type](parameter).fit(
                    train_vecs, train_labels
                )
                predicitions.append(model.predict(test_vecs))

            # get ensemble method performance metrics
            for ensemble_type in ensemble_types:
                if ensemble_type == 'average':
                    predicted_labels = np.round(np.mean(predicitions, axis=0))
                if ensemble_type == 'max_vote':
                    uniq, indices = np.unique(
                        predicitions, return_inverse=True
                    )
                    predicted_labels = uniq[np.argmax(np.apply_along_axis(
                        np.bincount, 0,
                        indices.reshape(np.array(predicitions).shape),
                        None, np.max(indices) + 1
                    ), axis=0)]
                if ensemble_type == 'weighted_average':
                    continue

                results[conversion][vector_size][ensemble_type] = {
                    label: classification_report(
                        test_labels[:, i], predicted_labels[:, i],
                        digits=3, zero_division=0, output_dict=True,
                        labels=label_classes[label]
                    )
                    for i, label in enumerate(labels)
                }

    # save results
    with open('data/results_ensemble.json', 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__':
    main()