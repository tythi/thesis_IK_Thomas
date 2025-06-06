import os
import sys
import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


def main():
    '''
    Prepares data by vectorising and converting all of the annotation according
     to a conversion
    '''
    # set environment hashseed to ensure non different vector inisialisations
    hashseed = os.getenv('PYTHONHASHSEED')
    if not hashseed:
        os.environ['PYTHONHASHSEED'] = '1'
        os.execv(sys.executable, [sys.executable] + sys.argv)

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

    conversion = 'medians'

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

    # get text from an annotators data
    common_texts = data[annotators[0]]['body'].to_numpy()

    # tag documents
    documents = [
        TaggedDocument(doc.split(), [i]) for i, doc in enumerate(common_texts)
    ]
    # intialise model
    vector_size = 50
    model = Doc2Vec(
        documents, vector_size=vector_size, min_count=1, workers=1, seed=1
    )
    # save model
    model.save(f'models/doc2vec_{vector_size}.model')

    # convert text to vectors
    vectors = data[annotators[0]]['body'].apply(
        lambda x: model.infer_vector(x.split())
    )

    # shift vectors to dodge negative values which dont work with MultinomialNB
    vectors = np.vstack(vectors)
    vectors = vectors + abs(vectors.min())
    print(vectors)

    # construct main dataframe
    main_data = pd.concat([
        data[annotators[0]]['name'],
        label_values,
    ], axis=1).rename(columns={'body': 'docvec'})

    pd.concat([
        data[annotators[0]]['name'],
        label_values,
        data[annotators[0]]['body']
    ], axis=1).to_csv(f'data/gold_standard_{conversion}.csv', index=False)

    # save vectors
    np.save(f'data/docvecs_{vector_size}.npy', np.vstack(vectors))

    # save data
    print(main_data)
    main_data.to_csv(f'data/label_{conversion}.csv', index=False)


if __name__ == '__main__':
    main()
