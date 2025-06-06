import os
import json
import jsonlines
import numpy as np
import pandas as pd
from tqdm import tqdm


def find_key_paths(data, target_key, path=None):
    '''
    Takes nested json files and returns route of keys from json to target in
     list of lists like:
        ```
            [
                ['comments', 2, 'body'],
                ['comments', 4, 'comments', 2, 'body'],
                ...
            ]
        ```
    '''
    if path is None:
        path = []

    found_paths = []

    if isinstance(data, dict):
        for key, value in data.items():
            new_path = path + [key]
            if key == target_key:
                found_paths.append(new_path)
            found_paths.extend(find_key_paths(value, target_key, new_path))

    elif isinstance(data, list):
        for index, item in enumerate(data):
            new_path = path + [index]
            found_paths.extend(find_key_paths(item, target_key, new_path))

    return found_paths


def get_value_by_path(data, path):
    '''
    By path like `['comments', 2, 'body']` get value of nested json
    '''
    for key in path:
        data = data[key]
    return data


def in_data_by_path(data, path, target):
    '''
    By path like `['comments', 2, 'body']` check if key: target is in nested
     json
    '''
    for key in path:
        data = data[key]
    return target in data


def get_data(data_file_paths=None, targets=None, type='json', nan_value=''):
    '''
    Get data from data_file_paths if specified, else get from
     /data/formatted_data
    Returns pandas dataframe with specified targets from given json or jsonl
     files
    '''
    if type not in ['json', 'jsonl']:
        raise TypeError('File format not supported')

    if not data_file_paths:
        data_dir_path = os.getcwd() + '/data/formatted_data'
        data_file_paths = os.listdir(data_dir_path)
    else:
        data_dir_path = os.getcwd() + '/data'

    if not all(path.endswith(('json', 'jsonl')) for path in data_file_paths):
        raise TypeError('File found with unsupported format')

    if not targets:
        targets = ['name', 'body']

    df = pd.DataFrame(columns=targets)

    count = 1

    for file_path in data_file_paths:
        if type == 'json':
            with open(f'{data_dir_path}/{file_path}') as f:
                data = json.load(f)
            paths = find_key_paths(data, 'body')
            for path in paths:
                df.loc[len(df)] = [
                    str(
                        get_value_by_path(data, path[:-1] + [target])
                    ).replace('\n', ' ').replace('\r', ' ')
                    if in_data_by_path(data, path[:-1], target)
                    else nan_value
                    for target in targets
                ]

        if type == 'jsonl':
            with jsonlines.open(f'{data_dir_path}/{file_path}') as reader:
                for data in tqdm(reader, desc='of 65169 these are done'):
                    # if df.shape[0] > 23000:
                    #     break
                    if count > 10:
                        break

                    paths = [['body']] + find_key_paths(data, 'body')
                    for path in paths:
                        df.loc[len(df)] = [
                            str(
                                get_value_by_path(data, path[:-1] + [target])
                            ).replace('\n', ' ').replace('\r', ' ')
                            if in_data_by_path(data, path[:-1], target)
                            else nan_value
                            for target in targets
                        ]
                    df.loc[0]['body'] = (
                        data['selftext'].replace('\n', ' ').replace('\r', ' ')
                    )
                    count += 1

    return df


def get_batch_indeces(a_list, batch_size=120):
    return [
        [index[0], index[-1] + 1] for index
        in np.array_split(range(len(a_list)), len(a_list) // batch_size)
    ]


def main():
    '''
    Extracts data from train/test dataset. Slightly obsolete code, superseded
     by jsonl.py
    Extracts data and relevant fields
    '''
    # columns to be in csv + targets
    annotation_targets = [
        'name', 'agency', 'event_sequencing', 'world_making', 'opt_comment',
        'story', 'body'
    ]

    nan_value = ''

    # json
    # targets = [
    #     'name', 'parent_id', 'created_utc', 'url', 'num_comments', 'author',
    #     'title', 'body'
    # ]
    # df = get_data(targets=targets, type='json', nan_value=nan_value)

    # jsonl
    data_file_paths = ['threads.jsonl']
    targets = (
        [
            'id'
        ]
        # +
        # [
        #     'parent_id', 'created_utc', 'url', 'num_comments', 'delta',
        #     'author', 'level', 'gilded', 'ups', 'downs', 'controversiality',
        #     'score', 'title'
        # ]
        +
        [
            'body'
        ]
    )
    df = get_data(
        targets=targets, data_file_paths=data_file_paths, type='jsonl',
        nan_value=nan_value
    )

    # remove footnote
    df['body'] = df['body'].replace(
        '_____  &gt; \*Hello, users of CMV!.*\*Happy CMVing!\*',
        '', regex=True
    )

    # remove rows with unwanted values
    df = df[~df['body'].isin(['[deleted]', '', '[redacted]', '[removed]'])]
    # df = df[df['author'] != 'DeltaBot']
    df = df[~df['title'].astype(str).str.startswith('TCMV')]
    df = df[~df['body'].astype(str).str.startswith((
        '**Most Popular Comments', '**Histogram -', '**Gilded Comments'))]
    df = df[~df['body'].astype(str).str.endswith((
        '\\))', 'at the request of this sub\'s moderators.)'))]

    df[df['author'] == 'DeltaBot']['body']

    # add remaining columns
    # for target in annotation_targets:
    #     if target not in targets:
    #         df.insert(len(df.columns) - 1, target, nan_value)
    # df = df[annotation_targets]

    # unselect unwanted rows
    # not_these = [
    #     't3_1i1ad3', 't1_cb00f3y', 't1_cb02fqy', 't1_cb00maq', 't1_cb03fvb',
    #     't1_cb04b9s', 't1_cb078iv', 't1_cb08ugs', 't1_cb0aoru', 't1_cb0gwh3',
    #     't1_cb0gzw9', 't1_cb0h4o1', 't1_cb1ac4j', 't1_cb1c1mr', 't1_cb1c37d',
    #     't1_cb1c3zk', 't3_2xi5bb', 't1_cp0ambd', 't1_cp0bshb', 't1_cp0c4jc',
    #     't1_cp0h7f5', 't1_cp0nthy', 't1_cp0qr6u', 't1_cp1c22n', 't1_cp0c6kp',
    #     't1_cp0f8wq', 't1_cp0gnbh'
    # ]
    # df = df[~df['name'].isin(not_these)]

    # take random subset
    # np.random.seed(0)
    # random_names = np.random.choice(df['name'], 620, replace=False)
    # df = df[df['name'].isin(random_names)]

    # together = [
    #     't1_cm5lvxq', 't1_cm5mup4', 't1_cm5xfbn', 't1_cm64nte', 't1_cm5mwyw',
    #     't1_cm5nasq', 't1_cm5niv6', 't1_cm5nwf0', 't1_cm5nyuj', 't3_1pg4zf',
    #     't1_cd2oaft', 't1_cd2rjoh', 't1_cd2ssov', 't3_2x4qkh', 't1_cowyuji',
    #     't1_cg5uill', 't1_cg5u5za', 't1_cg5vvtk', 't1_cg676fl', 't1_cg6891i',
    #     't1_cs8qzp2', 't1_cs8wghl', 't1_cs8ywso', 't1_cs96nzv', 't3_2cy7r8',
    #     't1_cjkhajj', 't1_cjkh0e7', 't1_cr706at', 't1_cr71uyn', 't1_c9vagau',
    #     't3_34yo0h', 't1_cqze8zq', 't1_cqzdpbc', 't1_chsthjv', 't1_chstlop',
    #     't1_chstdo9', 't1_chstnie', 't1_chtmsji', 't1_cf6guot', 't1_cf6jm5k',
    #     't1_cf6qwoo', 't1_clik16y', 't1_clikadz', 't1_clipfjz', 't1_cliyfh1',
    #     't1_cstu94m', 't1_csuck5d', 't1_cstvbko', 't1_cstvlvt', 't1_cstwvof',
    #     't1_csudcox', 't1_csvrg7x', 't1_cstvopg', 't1_cstx7qd', 't1_csuddim',
    #     't1_cstvpcf', 't1_cstvpqw', 't1_csu57ld', 't1_csu7lg5', 't1_csu8gdo',
    #     't1_csudgkr', 't1_csu0f6s', 't1_csv5ef5', 't1_csw1xe2', 't1_csu1ujy',
    #     't1_csu57h5', 't1_csudikx', 't3_1bcfz2', 't1_c95qp1r', 't1_ctl8trd',
    #     't1_ctlr9v3', 't1_ctlv7xc', 't1_ctltslz', 't1_cb3jjs2', 't1_cb3kir7',
    #     't1_cb3nynv', 't1_cb3qcwp', 't1_cb3tns6', 't1_cb3zy7m', 't1_cb4mafq',
    #     't1_cb4n4oh', 't1_cb46pb4', 't1_cb3qdsx', 't1_cb3un1d', 't1_cb3q90x',
    #     't1_cb3vnb9', 't1_cbcwihs', 't1_cb4019l', 't1_cb49a96', 't1_cbcwm6p',
    #     't3_1r3id3', 't3_1fqssq', 't1_cacvan5', 't1_cacvmf2', 't1_caf1ab9',
    #     't3_1qm5d4', 't1_cde6meq', 't1_cde6orn', 't1_cdeaz71', 't1_cdekndg',
    #     't3_1c4do4', 't1_c9cwx8n', 't1_c9dla74', 't1_c9d39gw', 't1_c9d5s0t',
    #     't1_cc5kmq0', 't1_cml0lai', 't1_cml2ft5', 't1_cml53oh', 't1_cml60ij',
    #     't1_cthvpeo', 't1_ctiqpyy', 't1_ctita17', 't1_ctisowf', 't1_ckxzlnz',
    #     't1_ckxzyeu', 't1_cky1afp', 't1_cky0d05', 't1_cky0fhe', 't1_cky0hj9',
    #     't1_cky2oax'
    # ]

    # together = df[df['name'].isin(together)]
    # together['body'] = together['body'].str.split()
    # print(together['body'].apply(len).sum())

    # save
    print(df)
    outfile = 'data/test.csv'
    df.to_csv(outfile, index=False)
    print(f'Saved to {outfile}')


if __name__ == '__main__':
    main()
