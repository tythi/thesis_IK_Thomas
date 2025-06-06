import krippendorff
import pandas as pd


def calculate_alphas(
        data, measurements, annotators, pretty_print=True, spaces_offset=4):
    alphas = {
        measurement: krippendorff.alpha([
            data[annotator][measurement].tolist()
            for annotator in annotators
        ])
        for measurement in measurements
    }
    alphas['average'] = sum(alphas.values()) / len(alphas.values())

    if pretty_print:
        offset = len(max(measurements, key=len))
        print(f'{" " * (spaces_offset - 4)}Krippendorff\'s alphas:')
        for key, value in alphas.items():
            print(f'{" " * spaces_offset}{key}: {" " * (offset - len(key))}'
                  + f'{value:.3f}')

    return alphas


def print_measurements(data, measurements, annotators, spaces_offset=4):
    offset = len(max(annotators, key=len))
    for measurement in measurements:
        print(f'{" " * (spaces_offset - 4)}{measurement} values:')
        for annotator in annotators:
            values = [
                int(value) for value in data[annotator][measurement].tolist()
            ]
            print(
                f'{" " * spaces_offset}{annotator}: '
                + f'{" " * (offset - len(annotator))}{values}'
            )


def main():
    '''
    Calculates Krippendorff's kappas for the target labels
    '''
    annotators = ['jani', 'manon', 'thomas']
    data = {
        annotator: pd.read_csv(
            'data/annotation_data/final_' + annotator + '.csv'
        ) for annotator in annotators
    }

    # first = [
    #     't3_1i1ad3', 't1_cb00f3y', 't1_cb02fqy', 't1_cb00maq', 't1_cb03fvb',
    #     't1_cb04b9s', 't1_cb078iv', 't1_cb08ugs', 't1_cb0aoru', 't1_cb0gwh3',
    #     't1_cb0gzw9', 't1_cb0h4o1', 't1_cb1ac4j', 't1_cb1c1mr', 't1_cb1c37d',
    #     't1_cb1c3zk'
    # ]
    # second = [
    #     't3_2xi5bb', 't1_cp0ambd', 't1_cp0bshb', 't1_cp0c4jc', 't1_cp0h7f5',
    #     't1_cp0nthy', 't1_cp0qr6u', 't1_cp1c22n', 't1_cp0c6kp', 't1_cp0f8wq',
    #     't1_cp0gnbh'
    # ]
    # data_targets = first + second

    data_targets = [
        't1_c95k50u', 't1_c95k75v', 't1_c95mz3n', 't1_c95tixb', 't1_c95l4my',
        't1_c95mcms', 't1_c95mdhe', 't3_1aenyc', 't1_c8wpqjn', 't1_c8wrovj',
        't1_c8wuxh2', 't1_cssxisl', 't1_c9zukmv', 't1_cjy73pb', 't1_cjy865a',
        't1_cjydv04', 't1_cjy9w7i', 't1_cg9ansq', 't1_cg7ybnn', 't1_cm5lbyg',
        't1_cm5lvxq', 't1_cm5mup4', 't1_cm5xfbn', 't1_cm64nte', 't1_cm5mwyw',
        't1_cm5nasq', 't1_cm5niv6', 't1_cm5nwf0', 't1_cm5nyuj', 't3_1pg4zf',
        't1_cd2oaft', 't1_cd2rjoh', 't1_cd2ssov', 't3_2x4qkh', 't1_cowyuji',
        't1_cg5uill', 't1_cg5u5za', 't1_cg5vvtk', 't1_cg676fl', 't1_cg6891i',
        't1_cs8qzp2', 't1_cs8wghl', 't1_cs8ywso', 't1_cs96nzv', 't3_2cy7r8',
        't1_cjkhajj', 't1_cjkh0e7', 't1_cr706at', 't1_cr71uyn', 't1_c9vagau',
        't3_34yo0h', 't1_cqze8zq', 't1_cqzdpbc', 't1_chsthjv', 't1_chstlop',
        't1_chstdo9', 't1_chstnie', 't1_chtmsji', 't1_cf6guot', 't1_cf6jm5k',
        't1_cf6qwoo', 't1_clik16y', 't1_clikadz', 't1_clipfjz', 't1_cliyfh1',
        't1_cstu94m', 't1_csuck5d', 't1_cstvbko', 't1_cstvlvt', 't1_cstwvof',
        't1_csudcox', 't1_csvrg7x', 't1_cstvopg', 't1_cstx7qd', 't1_csuddim',
        't1_cstvpcf', 't1_cstvpqw', 't1_csu57ld', 't1_csu7lg5', 't1_csu8gdo',
        't1_csudgkr', 't1_csu0f6s', 't1_csv5ef5', 't1_csw1xe2', 't1_csu1ujy',
        't1_csu57h5', 't1_csudikx', 't3_1bcfz2', 't1_c95qp1r', 't1_ctl8trd',
        't1_ctlr9v3', 't1_ctlv7xc', 't1_ctltslz', 't1_cb3jjs2', 't1_cb3kir7',
        't1_cb3nynv', 't1_cb3qcwp', 't1_cb3tns6', 't1_cb3zy7m', 't1_cb4mafq',
        't1_cb4n4oh', 't1_cb46pb4', 't1_cb3qdsx', 't1_cb3un1d', 't1_cb3q90x',
        't1_cb3vnb9', 't1_cbcwihs', 't1_cb4019l', 't1_cb49a96', 't1_cbcwm6p',
        't3_1r3id3', 't3_1fqssq', 't1_cacvan5', 't1_cacvmf2', 't1_caf1ab9',
        't3_1qm5d4', 't1_cde6meq', 't1_cde6orn', 't1_cdeaz71', 't1_cdekndg',
        't3_1c4do4', 't1_c9cwx8n', 't1_c9dla74', 't1_c9d39gw', 't1_c9d5s0t',
        't1_cc5kmq0', 't1_cml0lai', 't1_cml2ft5', 't1_cml53oh', 't1_cml60ij',
        't1_cthvpeo', 't1_ctiqpyy', 't1_ctita17', 't1_ctisowf', 't1_ckxzlnz',
        't1_ckxzyeu', 't1_cky1afp', 't1_cky0d05', 't1_cky0fhe', 't1_cky0hj9'
    ]

    label_targets = [
        'name', 'agency', 'event_sequencing', 'world_making', 'opt_comment',
        'story'
    ]

    for annotator in annotators:
        data[annotator] = data[annotator][label_targets].dropna(how='all')
        # data[annotator] = data[annotator][data[annotator]['name'].isin(
        #     data_targets)]

    measurements = ['agency', 'event_sequencing', 'world_making', 'story']
    for pair in [['jani', 'manon'], ['jani', 'thomas'], ['thomas', 'manon']]:
        print(pair)
        alphas = calculate_alphas(data, measurements, pair, pretty_print=True)
    print('All annotators')
    alphas = calculate_alphas(
        data, measurements, annotators, pretty_print=True
    )

    for annotator in annotators:
        print(annotator)
        print(data[annotator].value_counts('world_making'))
        print(round(data[annotator]['world_making'].mean(), 2))

    # print_measurements(data, measurements, annotators)


if __name__ == '__main__':
    main()
