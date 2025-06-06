import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm


def main():
    '''
    For each combination, print the best scoring models and the averages
     for each model type
    '''
    for conversion in ['medians', 'binary']:
        for vector_size in [50, 100, 200]:
            path = f'perfs_{vector_size}_{conversion}'
            print(f'Running on vector size {vector_size} and {conversion}')
            if f'{path}.csv' not in os.listdir(
                os.getcwd() + '/data'
            ):
                with open(
                    f'data/results_{vector_size}_{conversion}.json', 'r'
                ) as f:
                    data = json.load(f)

                model_perfs = pd.DataFrame(
                    columns=[
                        'model', 'parameter', 'macro_avg_f1-score',
                        'weighted_avg_f1-score'
                    ]
                )
                for model in data:
                    for parameter in tqdm(data[model], desc=f'Read {model}'):
                        model_perfs.loc[len(model_perfs)] = [
                            model, parameter,
                            sum([
                                (
                                    data[model][parameter][label]['macro avg']
                                    ['f1-score']
                                )
                                for label in data[model][parameter]
                            ]) / len(data[model][parameter]),
                            sum([
                                (
                                    data[model][parameter][label]
                                    ['weighted avg']['f1-score']
                                )
                                for label in data[model][parameter]
                            ]) / len(data[model][parameter])
                        ]

                model_perfs.to_csv(
                    f'data/{path}.csv', index=False
                )
            else:
                model_perfs = pd.read_csv(
                    os.getcwd() + f'/data/{path}.csv'
                )

            for measurement in ['macro_avg_f1-score', 'weighted_avg_f1-score']:
                print(f'Going for {measurement}')
                print(model_perfs.loc[
                    model_perfs.groupby('model')[measurement].idxmax()
                ])

                model_avgs = pd.DataFrame(
                    columns=['model', measurement]
                )

                for model in model_perfs['model'].unique():
                    model_avgs.loc[len(model_avgs)] = [
                        model,
                        (
                            model_perfs[model_perfs['model'] == model]
                            [measurement].mean()
                        )
                    ]

                print('Model averages:')
                print(model_avgs)

                show_graph = False

                model_perfs[measurement] = (
                    model_perfs[measurement].astype(float)
                )

                plt.figure(figsize=(12, 6))
                sns.boxplot(data=model_perfs, x='model', y=measurement)
                plt.xticks(rotation=45)
                plt.title(
                    f'Model performance (vector size: {vector_size}) '
                    f'(conversion: {conversion}) (average {measurement})'
                )
                plt.grid(axis='y')
                plt.tight_layout()
                plt.savefig(f'data/{path}_{measurement}.png')
                if show_graph:
                    plt.show()


if __name__ == '__main__':
    main()
