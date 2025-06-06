import math


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


def main():
    '''
    Get all parameter combinations for total count
    '''
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

    # combina all parameter combinations
    model_parameters = {model: [] for model in parameter_lists}
    for model in model_parameters:
        model_parameters[model] = combine_combinations(
            sorted(parameter_lists[model], key=str)
        )

    # math wise get total
    goal = sum(
        sum(
            math.prod([len(alist) for alist in parameters])
            for parameters in parameter_lists[model]
        )
        for model in parameter_lists
    )

    # compare
    for model in model_parameters:
        print(f'{model} {len(model_parameters[model])}')
    actual = sum(
        len(model_parameters[model])
        for model in model_parameters
    )
    print(goal == actual)
    print(f'{actual} parameter combinations')


if __name__ == '__main__':
    main()
