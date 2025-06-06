import json
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report


def main():
    '''
    Get performance metrics for baseline (most frequent) for both label
     conversions
    Gets most frequent occuring class for each label from the test set (first
     434), finds the most frequent class and generates a numpy array containing
     those most frequent labels against which the true labels are measured
    '''
    for conversion in ['medians', 'binary']:
        # initialise labels
        labels = ['agency', 'event_sequencing', 'world_making', 'story']

        # load corresponding data
        data = pd.read_csv(f'data/label_{conversion}.csv')

        # initialise unique classes for metrics
        label_classes = {
            label: data[label].unique() for label in labels
        }

        # get true labels
        true_labels = data[labels].to_numpy()

        # "predict" classes based on which is most frequent
        frequent_labels = np.column_stack(list((
            np.zeros(data.shape[0]) + data[label][:434].mode()[0]
            for label in labels
        )))

        # get test set
        true_labels, frequent_labels = true_labels[434:], frequent_labels[434:]

        # get performance metrics
        baseline_results = {}
        for i, label in enumerate(labels):
            baseline_results[label] = classification_report(
                true_labels[:, i], frequent_labels[:, i],
                digits=3, zero_division=0, output_dict=True,
                labels=label_classes[label]
            )

        # save data
        with open(f'data/baseline_results_{conversion}.json', 'w') as f:
            json.dump(baseline_results, f, indent=4)

        # "pretty" print stuff
        print(
            f'{conversion}' + ' ' * (7 - len(conversion)),
            ' labels    macro avg  weighted avg'
        )
        for label in labels:
            offset = len(max(baseline_results.keys(), key=len))
            print(
                f'    {label}:{" " * (offset - len(label) + 1)}',
                f'{baseline_results[label]["macro avg"]["f1-score"]:.3f}',
                '       ',
                f'{baseline_results[label]["weighted avg"]["f1-score"]:.3f}'
            )
        print(
            f'    average:{" " * (offset - len("average") + 1)}',
            round(sum([
                baseline_results[label]['macro avg']['f1-score']
                for label in baseline_results.keys()
            ]) / len(baseline_results.keys()), 3),
            '       ',
            round(sum([
                baseline_results[label]['weighted avg']['f1-score']
                for label in baseline_results.keys()
            ]) / len(baseline_results.keys()), 3),
        )


if __name__ == '__main__':
    main()
