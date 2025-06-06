import pandas as pd
from scipy.stats import chi2_contingency


def run_chi_square(data_subset, var1, var2):
    '''
    Returns chi square test results
    '''
    contingency = pd.crosstab(data_subset[var1], data_subset[var2])
    if contingency.shape == (2, 2):  # Ensure binary variables
        chi2, p, dof, _ = chi2_contingency(contingency)
        return {'chi2': chi2, 'p_value': p, 'dof': dof}
    return {'chi2': None, 'p_value': None, 'dof': None}


def main():
    '''
    Adaptation of Manon Koonings code with permission
    '''
    data = pd.read_csv('data/threads1000_format_labeled.csv')

    # Define variable pairs to test
    pairs = [
        ('story', 'agency'),
        ('story', 'event_sequencing'),
        ('story', 'world_making'),
        ('deltas', 'agency'),
        ('deltas', 'event_sequencing'),
        ('deltas', 'world_making'),
        ('deltas', 'story'),
    ]

    # Convert all values > 1 in deltas to 1 -> since a comment can receive
    # multiple deltas
    data['deltas'] = data['deltas'].apply(
        lambda x: 1 if pd.notna(x) and x > 0 else 0
    )

    binary_columns = [
        'deltas', 'story', 'agency', 'event_sequencing', 'world_making'
    ]
    data[binary_columns] = data[binary_columns].astype('Int64')

    # Run tests on whole dataset
    print('Chi-Square Tests on Entire Dataset:')
    for var1, var2 in pairs:
        result = run_chi_square(data, var1, var2)
        print(
            f'{var1} vs {var2}: χ² = {result["chi2"]:.3f}, p = '
            + f'{result["p_value"]:.3g}, dof = {result["dof"]}'
        )


if __name__ == '__main__':
    main()
