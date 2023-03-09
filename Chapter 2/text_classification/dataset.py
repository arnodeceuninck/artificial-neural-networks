from config import *
import pandas as pd
import torch
import random
random.seed(0)
import re


def tsv_to_list(tsv_file_path):
    df = pd.read_csv(tsv_file_path, sep='\t')
    texts = df[df.columns[0]].values.tolist()
    df[df.columns[1]] = df[df.columns[1]].replace(labels_mapper)
    labels = df[df.columns[1]].values.tolist()
    return texts, labels


def data_split(texts, labels, split_ratio=0.8):
    # split train data into train and validation
    tuples = list(zip(texts, labels))
    random.shuffle(tuples)
    data_length = len(tuples)
    tuples_train = tuples[:int(split_ratio*data_length)]
    tuples_val = tuples[int(split_ratio*data_length):]
    train_text, train_labels = zip(*tuples_train)
    val_text, val_labels = zip(*tuples_val)
    return train_text, train_labels, val_text, val_labels


# You can skip this part for now if you do not have an idea
def step1(texts):
    # hint: cleaning
    cleaned_texts = []
    for text in texts:
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        cleaned_texts.append(text)

    return cleaned_texts



def step2(texts):
    # hint: having a look-up-table
    # For each word in the vocabulary, we need to assign a unique index.
    # This is the purpose of the look-up-table.

    index = {'<unk>': 0, '<pad>': 1}
    for text in texts:
        for word in text.split():
            if word not in index:
                index[word] = len(index)
    return index

def build_vocab(texts):
    #code from assistent
    vocabs = list()
    additional_vocabs = ['<pad>', '<unk>']
    for text in texts:
        vocabs += text.split()
    vocabs = list(set(vocabs))
    vocabs.sort()
    vocabs = {word: index for index, word in enumerate(vocabs)}
    additional_vocabs.update(vocabs)
    return additional_vocabs

def step3(texts, vocab):
    # hint: converting strings to the required data format for an MLP

    # convert each word in text to a vector with numbers of the word in the vocabulary
    vectors = []
    for text in texts:
        vector = []
        for word in text.split():
            if word in vocab:
                vector.append(vocab[word])
            else:
                vector.append(vocab['<unk>'])
        vectors.append(vector)
    return vectors



def step4():
    # Something is missing ;)
    pass


def create_dataset(texts, labels, vocabs=None, max_length=50):
    # call all defined functions in sequence

    # step1()
    # .....
    # vectors, labels = step4()

    # print out each function to see the resulted outputs
    texts = step1(texts)

    # step2()
    vocab = step2(texts)

    # step3()
    mlp_encoded = step3(texts, vocab)
    print(mlp_encoded[0])

    # get the longest vector in mlp_encoded
    max_length = 97#max([len(vector) for vector in mlp_encoded])
    print(f"Own max length: {max_length}")
    mlp_encoded = padding(mlp_encoded, max_length, vocab)

    vectors = mlp_encoded

    vectors = torch.Tensor(vectors)
    labels = torch.Tensor(labels)
    return vectors, labels, vocabs

def padding(vectors, max_length, vocab):
    # hint: padding
    padded_vectors = []
    for vector in vectors:
        if len(vector) < max_length:
            pad_id = vocab['<pad>']
            vector.extend([pad_id]*(max_length-len(vector)))
        elif len(vector) > max_length:
            vector = vector[:max_length]
        padded_vectors.append(vector)
    return padded_vectors


if __name__ == "__main__":
    texts, labels = tsv_to_list(train_path)
    texts_train, labels_train, texts_val, labels_val = data_split(texts, labels, split_ratio=0.8)

    print(texts_train[0])

    # print out each function to see the resulted outputs
    texts_train = step1(texts_train)
    print(texts_train[0])
    texts_val = step1(texts_val)

    # step2()
    vocab = step2(texts_train)

    # step3()
    mlp_encoded = step3(texts_train, vocab)
    print(mlp_encoded[0])


    # get the longest vector in mlp_encoded
    max_length = max([len(vector) for vector in mlp_encoded])
    mlp_encoded = padding(mlp_encoded, max_length, vocab)

    texts = mlp_encoded

    # uncomment below lines after completing all functions
    texts, labels, vocabs = create_dataset(texts_train, labels_train, max_length=max_length)
    print(texts.shape)
    print(labels.shape)
