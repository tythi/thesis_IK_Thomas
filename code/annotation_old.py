import os
import json
import pandas as pd


def find_value_paths(data, target, path=None):
    if path is None:
        path = []

    found_paths = []

    if isinstance(data, dict):
        for key, value in data.items():
            new_path = path + [key]
            found_paths.extend(find_value_paths(value, target, new_path))

    elif isinstance(data, list):
        for index, item in enumerate(data):
            new_path = path + [index]
            found_paths.extend(find_value_paths(item, target, new_path))

    else:
        if data == target:
            found_paths.append(path)

    return found_paths


def find_key_paths(data, target_key, path=None):
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
    """Retrieve the value from a nested dictionary/list using a path."""
    for key in path:
        data = data[key]  # Navigate through the structure dynamically
    return data


def in_data_by_path(data, path, target):
    for key in path:
        data = data[key]
    return target in data


def print_json_values(data):
    for key, value in data.items():
        if isinstance(value, str) or isinstance(value, int):
            print(', '.join([key, value]))
        elif isinstance(value, dict):
            print_json_values(value)
        elif isinstance(value, list):
            for item in value:
                print_json_values(item)


def main():
    '''
    Old version to prepare the data for annotation
    '''
    data_dir_path = '/data/formatted_data'
    cwd = os.getcwd()

    data_file_paths = os.listdir(cwd + data_dir_path)

    # with open(cwd + data_dir_path + '/' + data_file_paths[0]) as f:
    #     data = json.load(f)

    # total_deltas = 0
    # for path in data_file_paths:
    #     with open(cwd + data_dir_path + '/' + path) as f:
    #         data = json.load(f)
    #     total_deltas += len(find_value_paths(data, 'DeltaBot'))

    # paths = find_value_paths(data, 'DeltaBot')
    # print(total_deltas)

    # for path in data_file_paths:
    #     with open(cwd + data_dir_path + '/' + path) as f:
    #         data = json.load(f)
    #         if data['title'][:3] != 'CMV' and data['title'][:2] != 'I ':
    #             print(f'{path} {data["title"][:40]}')

    # for path in data_file_paths[:2]:
    #     with open(cwd + data_dir_path + '/' + data_file_paths[0]) as f:
    #         data = json.load(f)
    #         print_json_values(data)

    unique_keys = ['author', 'body', 'comments', 'created_utc', 'name',
                   'num_comments', 'parent_id', 'replies', 'title', 'url']
    targets = ['name', 'parent_id', 'title', 'author', 'num_comments',
               'created_utc', 'url', 'body']
    targets = ['name', 'agency', 'event_sequencing', 'world_making',
               'opt_comment', 'story', 'body']
    count = 0
    nan_value = ''

    df = pd.DataFrame(columns=targets)

    print('@'.join(targets))
    for file_path in data_file_paths:
        with open(cwd + data_dir_path + '/' + file_path) as f:
            data = json.load(f)
            paths = find_key_paths(data, 'body')
            for path in paths:
                if any([
                    get_value_by_path(
                        data, path[:-1] + ['body']
                    ) == '[deleted]',
                    get_value_by_path(data, path[:-1] + ['body']) == '',
                    get_value_by_path(
                        data, path[:-1] + ['author']
                    ) == 'DeltaBot'
                ]):
                    continue
                if in_data_by_path(data, path[:-1], 'title'):
                    if get_value_by_path(
                        data, path[:-1] + ['title']
                    )[:4] == 'TCMV':
                        continue
                to_print = [
                    str(get_value_by_path(data, path[:-1] + [target])).replace(
                        '\n', ' '
                    )
                    if in_data_by_path(data, path[:-1], target)
                    else nan_value
                    for target in targets
                ]
                if set(to_print) != {nan_value}:
                    print('@'.join(to_print))


if __name__ == '__main__':
    main()
