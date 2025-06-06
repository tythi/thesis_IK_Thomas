import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB


def set_combinations(lists):
    if not lists:
        return {()}

    results = set()

    for item in lists[0]:
        for combo in set_combinations(lists[1:]):
            results.add((item,) + combo)

    return results


def combine_combinations(parameter_lists):
    results = [
        set_combinations(parameters)
        for parameters in parameter_lists
    ]
    return list(set().union(*results))


def evaluate_model(
        model, params, train_vecs, train_labels, test_vecs, test_labels,
        labels, label_classes
):
    model.fit(train_vecs, train_labels)
    predicted_labels = model.predict(test_vecs)

    label_results = {
        label: classification_report(
            test_labels[:, i], predicted_labels[:, i],
            digits=3, zero_division=0, output_dict=True,
            labels=label_classes[label]
        )
        for i, label in enumerate(labels)
    }

    return str(params), label_results


def main():
    # set seeds
    np.random.seed(1)
    random.seed(1)

    # select conversion
    conversion = 'medians'
    data = pd.read_csv(f'data/label_{conversion}.csv')

    labels = ['agency', 'event_sequencing', 'world_making', 'story']

    # initialise unique classes for metrics
    label_classes = {
        label: data[label].unique() for label in labels
    }

    # select vector size
    true_labels = (data.dropna(how='any')[labels].to_numpy())
    vector_size = 50

    # load data
    text_vectors = np.load(f'data/docvecs_{vector_size}.npy')
    text_vectors = text_vectors[:true_labels.shape[0]]
    train_vecs, test_vecs, train_labels, test_labels = train_test_split(
        text_vectors, true_labels, test_size=0.3, shuffle=False
    )

    # initialise parameter combinations
    parameter_lists = {
        'svc': [
            # C
            # kernel
            # degree
            # gamma
            # coef0
            [
                [0.5, 0.75, 1.0, 1.25, 1.5],
                ['sigmoid'],
                [3],
                ['scale', 'auto'] + [0.15, 0.2, 0.25, 0.3, 0.35],
                [-0.5, 0.0, 0.5]
            ],
            [
                [0.5, 0.75, 1.0, 1.25, 1.5],
                ['poly'],
                [1, 2, 3, 4, 5],
                ['scale', 'auto'] + [0.15, 0.2, 0.25, 0.3, 0.35],
                [-0.5, 0.0, 0.5]
            ],
            [
                [0.5, 0.75, 1.0, 1.25, 1.5],
                ['rbf'],
                [3],
                ['scale', 'auto'] + [0.15, 0.2, 0.25, 0.3, 0.35],
                [0]
            ],
            [
                [0.5, 0.75, 1.0, 1.25, 1.5],
                ['linear'],
                [3],
                ['scale'],
                [0]
            ]
        ],
        'knn': [
            # n_neighbors
            # weights
            # algorithm
            # leaf_size
            # p
            [
                [3, 4, 5, 6, 7],
                ['uniform', 'distance'],
                ['auto', 'ball_tree', 'kd_tree', 'brute'],
                [20, 25, 30, 35, 40],
                [1.5, 1.75, 2.0, 2.25, 2.5]
            ]
        ],
        'random forest': [
            # n_estimators
            # criterion
            # min_samples_split
            # min_samples_leaf
            # max_features
            # min_impurity_decrease
            # ccp_alpha
            [
                [150, 175, 200, 225, 250],
                ['gini', 'entropy', 'log_loss'],
                [1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5],
                ['sqrt', 'log2', None],
                [0, 0.25, 0.5],
                [0.0, 0.0001, 0.001, 0.01, 0.1],
            ]
        ],
        'multinomialnb': [
            # alpha
            # force_alpha
            [
                [0.5, 0.75, 1, 1.25, 1.5],
                [True, False],
            ]
        ],
        'guassiannb': [
            # var_smoothing
            [
                [1e-7, 1e-8, 1e-9, 1e-10, 1e-11],
            ]
        ],
    }

    # combine parameter combinations
    model_parameters = {model: [] for model in parameter_lists}
    for model in model_parameters:
        model_parameters[model] = combine_combinations(
            sorted(parameter_lists[model], key=str)
        )

    # intitialise dict for model loading
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

    # initialise dict for results
    results = {model_type: {} for model_type in model_parameters}

    # run every model with every parameter combination with parallelisation
    print(f'Running on vector size {vector_size} and {conversion}')
    for model_type in model_parameters:
        processed = Parallel(n_jobs=-1)(
            delayed(evaluate_model)(
                models[model_type](params), params,
                train_vecs, train_labels, test_vecs, test_labels, labels,
                label_classes
            )
            for params in tqdm(model_parameters[model_type], desc=model_type)
        )

        # Store results
        for param_str, label_results in processed:
            results[model_type][param_str] = label_results

    with open(f'data/results_{vector_size}_{conversion}.json', 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__':
    main()
