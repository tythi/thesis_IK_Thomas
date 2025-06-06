import pandas as pd


def main():
    '''
    Preproceses a csv file according to the rules below
    '''
    file_path = 'data/threads1000_format.csv'
    df = pd.read_csv(file_path)

    print(df[['name', 'deltas', 'body']])

    # remove footnote
    df['body'] = df['body'].replace(
        '_____  &gt; \*Hello, users of CMV!.+\*Happy CMVing!\*',
        '', regex=True
    )

    # remove rows with unwanted values
    df = df[~df['body'].isin(['[deleted]', '[redacted]', '[removed]'])]
    df = df[df['author'] != 'DeltaBot']
    df = df[~df['title'].astype(str).str.startswith('TCMV')]
    df = df[~df['body'].astype(str).str.startswith((
        '**Most Popular Comments', '**Histogram -', '**Gilded Comments'))]
    df = df[
        ~df['body'].astype(str).str.endswith((
            '\\))', 'at the request of this sub\'s moderators.)'
        ))
    ]

    df['text'] = df['title'].fillna('') + ' ' + df['body'].fillna('')

    preprocessed = df[['name', 'deltas', 'text']].rename(
        columns={'text': 'body'}
    )
    print(preprocessed)

    outfile_path = 'data/threads1000_format_preprocessed.csv'
    preprocessed.to_csv(outfile_path, index=False)


if __name__ == '__main__':
    main()
