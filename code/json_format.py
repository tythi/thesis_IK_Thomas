import json


def to_pretty(this, that):
    with open(this, 'r') as f:
        data = json.load(f)
    with open(that, 'w') as f:
        json.dump(data, f, indent=4)


def no_pretty(this, that):
    with open(this, 'r') as f:
        data = json.load(f)
    with open(that, 'w') as f:
        json.dump(data, f)


def main():
    '''
    Reduces the file size of a json file by removing the indentation
    '''
    no_pretty('data/results_200.json', 'data/results_200_smaller.json')


if __name__ == '__main__':
    main()
