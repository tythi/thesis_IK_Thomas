import pandas as pd


def main():
    '''
    Code partially from prepare_data.py because it works for this too
    Reads annotated csv files, extracts the median values and binary values and
     displays their distributions
    '''
    # get annotation data
    annotators = ['jani', 'manon', 'thomas']
    data = {
        annotator: pd.read_csv(
            'data/annotation_data/final_' + annotator + '.csv'
        )
        for annotator in annotators
    }

    # prepare data targets
    label_targets = [
        'agency', 'event_sequencing', 'world_making', 'story'
    ]

    for conversion in ['medians', 'binary']:
        # intialise conversion data dataframe
        label_values = pd.DataFrame(columns=label_targets)

        if conversion == 'medians':
            # get median data
            for label in label_targets:
                label_values[label] = pd.concat([
                    data[annotator][label] for annotator in annotators
                ], axis=1).median(axis=1)
        elif conversion == 'binary':
            # get binary values
            for label in label_targets:
                row = pd.concat([
                    data[annotator][label] for annotator in annotators
                ], axis=1).mean(axis=1)
                label_values[label] = (
                    row.map(lambda x: 0 if x < 2.5 else 1)
                    if label in ['agency', 'event_sequencing', 'world_making']
                    else row.map(lambda x: round(x))
                )

        # print value distributions
        for label in label_targets:
            print(label_values[label].value_counts().sort_index())


if __name__ == '__main__':
    main()
