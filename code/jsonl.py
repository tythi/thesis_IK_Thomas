import os
import json
import jsonlines
import pandas as pd
from tqdm import tqdm


def get_field_dict(data, n):
    '''
    Returns relevant fields found in data and adds discussion number
    '''
    return {
        'archived': data.get('archived', bool()),
        'author_flair_text': data.get('author_flair_text', str()),
        'author': data.get('author', str()),
        'body': data.get('body', str()),
        'controversiality': data.get('controversiality', int()),
        'created_utc': data.get('created_utc', str()),
        'created': data.get('created', int()),
        'delta': data.get('delta', bool()),
        'deltas': int(),
        'discussion_number': n,
        'distinguished': data.get('distinguished', None),
        'domain': data.get('domain', str()),
        'downs': data.get('downs', int()),
        'edited': data.get('edited', int()),
        'from_id': data.get('from_id', None),
        'from_kind': data.get('from_kind', None),
        'from': data.get('from', None),
        'gilded': data.get('gilded', int()),
        'hide_score': data.get('hide_score', bool()),
        'id': data.get('id', str()),
        'is_self': data.get('is_self', bool()),
        'level': data.get('level', int()),
        'link_flair_text': data.get('link_flair_text', str()),
        'link_id': data.get('link_id', str()),
        'media_embed': data.get('media_embed', list()),
        'media': data.get('media', None),
        'name': data.get('name', str()),
        'num_comments': data.get('num_comments', int()),
        'over_18': data.get('over_18', bool()),
        'permalink': data.get('permalink', str()),
        'quarantine': data.get('quarantine', bool()),
        'removal_reason': data.get('removal_reason', None),
        'retrieved_on': data.get('retrieved_on', int()),
        'saved': data.get('saved', bool()),
        'score_hidden': data.get('score_hidden', bool()),
        'score': data.get('score', int()),
        'secure_media_embed': data.get('secure_media_embed', list()),
        'secure_media': data.get('secure_media', None),
        'selftext': data.get('selftext', str()),
        'stickied': data.get('stickied', bool()),
        'subreddit_id': data.get('subreddit_id', str()),
        'subreddit': data.get('subreddit', str()),
        'thumbnail': data.get('thumbnail', str()),
        'title': data.get('title', str()),
        'ups': data.get('ups', int()),
        'url': data.get('url', str()),
        'urls': data.get('urls', list()),
    }


def main():
    '''
    Extracts relevant fields and their data from given jsonl file. Supersedes
     extract_data.py because that was inefficient pandas
     dataframe shenanigans
    Saves (probably massive) csv and json files
    '''
    # initialise data dicts
    result = dict()
    parents = dict()
    data_file_path = os.getcwd() + '/data/threads.jsonl'
    count = 0
    # open jsonl
    with jsonlines.open(data_file_path) as reader:
        # nice progress bar
        for data in tqdm(reader):
            # optional count to stop after 'count' lines
            if count >= 65170:
                break

            # retrieve all relevant data fields
            result[data.get('name', '')] = get_field_dict(data, count)

            # get children
            comments = data.get('comments', [])

            def recurs_comments(comments):
                '''
                Recursively retrieve all relevant data fields from comments'
                 comment fields which contain more comments. Put inside of the
                 loop here because that makes it easier to write recursive
                 functions so it can just throw the data into a variable
                Does not return anything but in this case puts the data in
                 result
                '''
                for comment in comments:
                    result[comment.get('name', '')] = get_field_dict(
                        comment, count
                    )
                    parents[comment.get('name')] = comment.get('parent_id')

                    if (
                        comment.get('author', '') == 'DeltaBot'
                        and (
                            '1 delta awarded' in comment.get('body')
                            or '1 point awarded' in comment.get('body')
                        )
                    ):
                        if (parent := comment.get('parent_id')) in parents:
                            if parent in parents:
                                result[parents[parent]]['deltas'] += 1

                    recurs_comments(comment.get('children', []))

            recurs_comments(comments)

            count += 1

    outfile = 'data/threads_format'

    # save json files
    with open(outfile + '.json', 'w') as f:
        json.dump(list(result.values()), f, indent=4)
    with open(outfile + '_noindent.json', 'w') as f:
        json.dump(list(result.values()), f)
    print(f'Wrote to {outfile}.json')

    # save csv file
    df = pd.DataFrame(list(result.values()))
    del result
    print(df)
    df.to_csv(outfile + '.csv', index=False)


if __name__ == '__main__':
    main()
