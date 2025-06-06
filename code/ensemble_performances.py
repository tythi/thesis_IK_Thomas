import json
import pandas as pd


def main():
    '''
    Prints out ensemble performances in more readable format
    '''
    with open('data/results_ensemble.json', 'r') as f:
        data = json.load(f)

    ensemble_perfs = pd.DataFrame(
        columns=[
            'conversion', 'vector_size', 'ensemble_type', 'macro avg f1-score',
            'weighted avg f1-score'
        ]
    )

    for conversion in ['medians', 'binary']:
        for vector_size in ['50', '100', '200']:
            for ensemble_type in ['average', 'max_vote']:
                ensemble_perfs.loc[len(ensemble_perfs)] = [
                    conversion, vector_size, ensemble_type,
                    sum([
                        (
                            data[conversion][vector_size][ensemble_type][label]
                            ['macro avg']['f1-score']
                        )
                        for label
                        in data[conversion][vector_size][ensemble_type]
                    ]) / len(data[conversion][vector_size][ensemble_type]),
                    sum([
                        (
                            data[conversion][vector_size][ensemble_type][label]
                            ['weighted avg']['f1-score']
                        )
                        for label
                        in data[conversion][vector_size][ensemble_type]
                    ]) / len(data[conversion][vector_size][ensemble_type])
                ]

    print(ensemble_perfs)


if __name__ == '__main__':
    main()
