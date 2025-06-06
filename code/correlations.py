import pandas as pd
from scipy.stats import chi2_contingency


def main():
    '''
    Calculates chi-square correlations between structural labels and deltas,
     comparing short vs long discussions (by number of comments).
    '''
    data = pd.read_csv('data/threads1000_format_labeled.csv')

    # make deltas are binary
    data['deltas'] = data['deltas'].apply(
        lambda x: 1 if pd.notna(x) and x > 0 else 0
    )

    # make sure labels are binary
    labels = ['agency', 'event_sequencing', 'world_making', 'story']
    for label in labels:
        data[label] = data[label].astype(int)

    # remove irrelevant rows
    data = data[data['author'] != 'DeltaBot']

    # count number of comments per discussion
    discussion_lengths = data['discussion_number'].value_counts()
    data['discussion_length'] = data['discussion_number'].map(
        discussion_lengths
    )

    # define short and long discussions
    ranges = {
        'short discussions (< 20 comments)': data['discussion_length'] < 20,
        'long discussions (>= 20 comments)': data['discussion_length'] >= 20
    }

    for description, mask in ranges.items():
        print(f'\n--- {description} ---')
        subset = data[mask]

        for label in labels:
            ct = pd.crosstab(subset['deltas'], subset[label])

            if ct.shape == (2, 2):
                chi2, p, _, _ = chi2_contingency(ct)
                print(f'deltas vs {label}: χ² = {chi2:.3f}, p = {p:.3g}')
            else:
                print(f'deltas vs {label}: Not enough variation')


if __name__ == '__main__':
    main()
