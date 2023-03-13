import os
import re
import time

import numpy as np
import pandas as pd

"""Citirea datelor"""
data_path = 'D:\\FACULTATE\\ANUL 3\\INTELIGENTA ARTIFICIALA\\Proiect Kaggle\\ub-fmi-cti-2022-2023'
train_data_df = pd.read_csv(os.path.join(data_path, 'train_data.csv'))
test_data_df = pd.read_csv(os.path.join(data_path, 'test_data.csv'))

print('Distributia etichetelor in datele de antrenare: ')
print(train_data_df['label'].value_counts(), '\n')

print('Fiecare limba este reprezentata in mod egal:')
print(train_data_df['language'].value_counts())
print(train_data_df['label'].unique(), '\n')

"""Codificarea etichetelor din string in int"""
# codificam etichetele / labels in valori cu numere intregi dela 0 la N

etichete_unice = train_data_df['label'].unique()
label2id = {}
id2label = {}
for idx, eticheta in enumerate(etichete_unice):
    label2id[eticheta] = idx
    id2label[idx] = eticheta

print('Codificarea etichetelor')
print('Din eticheta in ID: ')
print(label2id)
print('Din ID in eticheta: ')
print(id2label, '\n')

# aplicam dictionarul label2id peste toate etichetele din train
labels = []
for eticheta in train_data_df['label']:
    labels.append(label2id[eticheta])
labels = np.array(labels)

labels = train_data_df['label'].apply(lambda etich: label2id[etich])
labels = labels.values

"""
Pre-procesarea datelor
- Extragerea informațiilor necesare din text
- Eliminarea semnelor de punctuație
- Impartirea in cuvinte (Tokenization)
"""


def proceseaza(text):
    # Functie simpla de procesare a textului
    text = re.sub("[-.,;:!?\"\'\/()_*=`]", "", text)
    text = text.replace('\n', ' ').strip().lower()
    text_in_cuvinte = text.split(' ')
    return text_in_cuvinte


# cuvintele rezultate din functia de preprocesare:
# exemple_italian = train_data_df[train_data_df['language'] == 'italiano']
# print(exemple_italian)

# text_italian = exemple_italian['text'].iloc[0]
# print(proceseaza(text_italian)[:13])

"""Aplicam functia de preprocesarea intregului set de date"""
data = train_data_df['text'].apply(proceseaza)
# print(data[0])
# print(type(data))
# print(data)

# atunci cand facem df.apply() obtinem un tip de date pd.Series care este indexabil dupa o lista la fel ca un np array
# print(data[[3,800,41569]])
# accesarea elementului de pe pozitia 0 din slice data[[3,800,41569]][0]
# se face asta doar prin .iloc data[[3,800,41569]].iloc[0]

"""Împărțirea datelor în train, validare și test"""
print('Impartirea datelor')
# O împărțeală brutală poate fi cea în funcție de ordinea în care apar datele
print('Nr total de date: ', len(data))
# print(len(data), '\n')

# putem imparti datele de antrenare astfel:
# 20% date de test din total
# 15% date de validare din ce ramane dupa ce scoatem datele de test

nr_test = int(20 / 100 * len(train_data_df))
print("Nr de date de test: ", nr_test)
nr_ramase = len(data) - nr_test
nr_valid = int(15 / 100 * nr_ramase)
print("Nr de date de validare: ", nr_valid)

nr_train = nr_ramase - nr_valid
print("Nr de date de antrenare: ", nr_train, '\n')

# luam niste indici de la 0 la N
indici = np.arange(0, len(train_data_df))
# print('Amestecarea datelor')
# print(indici)
# ii permutam si apoi putem sa-i folosim pentru a amesteca datele
np.random.shuffle(indici)
# print(indici, '\n')

# facem impartirea in ordinea in care apar datele
# datele se amesteca folosind indicii permutati, in loc de split in functie
# ordinea in care apar exemplele
train_data = data[indici[:nr_train]]
train_labels = labels[indici[:nr_train]]

valid_data = data[indici[nr_train: nr_train + nr_valid]]
valid_labels = labels[indici[nr_train: nr_train + nr_valid]]

test_data = data[indici[nr_train + nr_valid:]]
test_labels = labels[indici[nr_train + nr_valid:]]

print(f'Nr de exemple de train: {len(train_labels)}')
print(f'Nr de exemple de validare: {len(valid_labels)}')
print(f'Nr de exemple de test: {len(test_labels)}')

"""CountVectorizer"""
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(tokenizer=lambda x: x,  # data e deja procesat, nu mai e nevoie de tokenizer aici
                             preprocessor=lambda x: x,  # data e deja procesat, nu mai e nevoie de tokenizer aici
                             max_features=100000)
print('\nVectorizing...')
vectorizer.fit(train_data)
X_train = vectorizer.transform(train_data)
X_valid = vectorizer.transform(valid_data)
X_test = vectorizer.transform(test_data)

"""Antrenare SVM"""
from sklearn.metrics import accuracy_score
from sklearn import svm

model = svm.LinearSVC(C=0.1)

antrenare_start = time.time()

print('\nSVM - Fitting...')
model.fit(X_train, train_labels)

vpreds = model.predict(X_valid)
tpreds = model.predict(X_test)

duratie_antrenare = (time.time() - antrenare_start)
print('Durata antrenarii:', duratie_antrenare, 's.')

print('Acuratete pe validare ', accuracy_score(valid_labels, vpreds))
print('Acuratete pe test ', accuracy_score(test_labels, tpreds))

"""Procesam datele de test pentru a le vectoriza predictii"""
date_test_procesate = test_data_df['text'].apply(proceseaza)
date_test_vectorizate = vectorizer.transform(date_test_procesate)
predictii = model.predict(date_test_vectorizate)

""""Folosim pandas pentru a salva predictiile in format .csv"""
rezultat = pd.DataFrame({'id': np.arange(1, len(predictii) + 1), 'label': predictii})

# putem numi fisierul in functie de hiperparametrii si model
nume_model = str(model)
print('\nNume model: ', nume_model)
nr_de_caracteristici = 'N=100000'
print('Nr. de caracteristici: ', nr_de_caracteristici)
functie_preprocesare = 'count_vectorizer'
print('Functie preprocesare: ', functie_preprocesare)

nume_fisier = '_'.join([nume_model, nr_de_caracteristici, functie_preprocesare]) + '.csv'

# salvam rezultatul fara index intr-un fisier de tip csv
rezultat.to_csv(nume_fisier, index=False)
print('\nRezultatul a fost salvat in fisierul csv!')
print('Fisier: ', nume_fisier)


