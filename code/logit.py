import pandas as pd
import statsmodels.formula.api as smf


def print_label_distributions(dataframe):
    for label in ['agency', 'event_sequencing', 'world_making', 'story']:
        print(dataframe[label].value_counts().sort_index())


def main():
    '''
    Performs the logistic regression for both all of the structural elements
     and their interations, and time_since_ops interactions with each
     individual structural element
    '''
    data = pd.read_csv('data/threads1000_format_labeled.csv')

    targets = data[['agency', 'event_sequencing', 'world_making', 'story']]

    for col in ['agency', 'event_sequencing', 'world_making']:
        targets[col] = targets[col].astype('category')

    targets = pd.get_dummies(
        targets, columns=['agency', 'event_sequencing', 'world_making'],
        drop_first=True
    )

    # perform logistic regression
    model = smf.logit('story ~ ' + ' + '.join(
        targets.columns.difference(['story'])
    ), data=targets).fit()

    print(model.summary())

    # get time since op
    data = data[data['created_utc'] != 0]
    is_original = data['name'].fillna('').str.startswith('t3_')
    op_times = data[is_original].set_index('discussion_number')['created_utc']
    data['op_created_utc'] = data['discussion_number'].map(op_times)
    data['time_since_op'] = data['created_utc'] - data['op_created_utc']
    data = data[~is_original].copy()
    data = data.drop(columns=['op_created_utc'])

    data['deltas'] = data['deltas'].apply(
        lambda x: 1 if pd.notna(x) and x > 0 else 0
    )

    labels = ['agency', 'event_sequencing', 'world_making', 'story']

    # perform logistic regression
    targets = data[labels + ['deltas', 'time_since_op']]
    for label in labels:
        model = smf.logit(
            formula=f'deltas ~ {label} * time_since_op',
            data=data
        ).fit()

        print(model.summary())


if __name__ == '__main__':
    main()
