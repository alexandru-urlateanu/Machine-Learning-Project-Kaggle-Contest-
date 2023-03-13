import os
import re
import time
from collections import Counter

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

"""Bag of Words"""


# În cadrul acestei secțiuni vom face numărarea aparițiilor tuturor cuvintelor din datele noastre.
# Pentru o evaluare justă nu ar fi indicat să includem si cuvintele din datele de test.
def count_most_common(how_many, texte_preprocesate):
    # Functie care returneaza cele mai frecvente cuvinte
    counter = Counter()
    # texte pre-procesate
    for text in texte_preprocesate:
        # update peste o lista de cuvinte
        counter.update(text)
    cele_mai_frecvente = counter.most_common(how_many)
    cuvinte_caracteristice = [cuvant for cuvant, _ in cele_mai_frecvente]
    return cuvinte_caracteristice


def build_id_word_dicts(cuvinte_caracteristice):
    # Dictionarele word2id si id2word garanteaza o ordine pentru cuvintele caracteristice
    word2id = {}
    id2word = {}
    for idx, cuv in enumerate(cuvinte_caracteristice):
        word2id[cuv] = idx
        id2word[idx] = cuv
    return word2id, id2word


def featurize(text_preprocesat, id2word):
    """Pentru un text preprocesat dat si un dictionar
    care mapeaza pentru fiecare pozitie ce cuvant corespunde,
    returneaza un vector care reprezinta
    frecventele fiecarui cuvant.
    """
    # 1. numaram toate cuvintele din text
    ctr = Counter(text_preprocesat)

    # 2. prealocam un array care va reprezenta caracteristicile noastre
    features = np.zeros(len(id2word))

    # 3. umplem array-ul cu valorile obtinute din counter
    # fiecare pozitie din array trebuie sa reprezinte frecventa aceluiasi cuvant in toate textele
    for idx in range(0, len(features)):
        # obtinem cuvantul pentru pozitia idx
        cuvant = id2word[idx]
        # asignam valoarea corespunzatoare frecventei cuvantului
        features[idx] = ctr[cuvant]

    return features


def featurize_multi(texte, id2word):
    '''Pentru un set de texte preprocesate si un dictionar
    care mapeaza pentru fiecare pozitie ce cuvant corespunde,
    returneaza matricea trasaturilor tuturor textelor.
    '''
    all_features = []
    for text in texte:
        all_features.append(featurize(text, id2word))
    return np.array(all_features)


"""Transformam datele in format vectorial"""
cuvinte_caracteristice = count_most_common(1000, train_data)
# print(len(cuvinte_caracteristice))
word2id, id2word = build_id_word_dicts(cuvinte_caracteristice)

X_train = featurize_multi(train_data, id2word)
X_valid = featurize_multi(valid_data, id2word)
X_test = featurize_multi(test_data, id2word)

"""Facem experimente pe impartire in train-valid-test"""

"""Antrenare SVM"""
from sklearn.metrics import accuracy_score
from sklearn import svm

model = svm.LinearSVC(C=0.25)

antrenare_start = time.time()

print('\nSVM - Fitting...')
model.fit(X_train, train_labels)

vpreds = model.predict(X_valid)
tpreds = model.predict(X_test)

duratie_antrenare = (time.time() - antrenare_start)
print('Durata antrenarii:', duratie_antrenare, 's.')

print('Acuratete pe datele de validare:')
print(accuracy_score(valid_labels, vpreds))

print('Acuratete pe datele de test:')
print(accuracy_score(test_labels, tpreds))

"""
Reantrenam modelul
"""

toate_datele_vectorizate = featurize_multi(data, id2word)

# antrenam modelul pe toate datele
print('\nFitting pe toate datele...')
model.fit(toate_datele_vectorizate, train_data_df['label'])

# # vedem ce acuratete obtine pe datele pe care s-a antrenat?
# # daca modelul invata repartitia datelor, ar trebui sa fie o acuratete foarte buna
predictii_pe_train = model.predict(toate_datele_vectorizate)
print('Acuratete pe datele deja vazute de model:')
print(accuracy_score(train_data_df['label'], predictii_pe_train), '\n')

"""Procesam datele de test pentru a le vectoriza predictii"""
date_test_procesate = test_data_df['text'].apply(proceseaza)
date_test_vectorizate = featurize_multi(date_test_procesate, id2word)
predictii = model.predict(date_test_vectorizate)

""""Folosim pandas pentru a salva predictiile in format .csv"""
rezultat = pd.DataFrame({'id': np.arange(1, len(predictii) + 1), 'label': predictii})

# putem numi fisierul in functie de hiperparametrii si model
nume_model = str(model)
print('\nNume model: ', nume_model)
nr_de_caracteristici = f'N={len(id2word)}'
print('Nr. de caracteristici: ', nr_de_caracteristici)
functie_preprocesare = 'lower,strip,no_punct'
print('Functie preprocesare: ', functie_preprocesare)

nume_fisier = '_'.join([nume_model, nr_de_caracteristici, functie_preprocesare]) + '.csv'

# salvam rezultatul fara index intr-un fisier de tip csv
rezultat.to_csv(nume_fisier, index=False)
print('\nRezultatul a fost salvat in fisierul csv!')
print('Fisier: ', nume_fisier, '\n')

"""Cross Validare"""
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5)
toate_etichetele = train_data_df['label'].values
print(skf.get_n_splits(toate_datele_vectorizate, toate_etichetele))
for train_index, test_index in skf.split(toate_datele_vectorizate, toate_etichetele):
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train_cv, X_test_cv = toate_datele_vectorizate[train_index], toate_datele_vectorizate[test_index]
    y_train_cv, y_test_cv = toate_etichetele[train_index], toate_etichetele[test_index]
    # print(X_train_cv.shape)
    model = svm.LinearSVC(C=0.25)
    model.fit(X_train_cv, y_train_cv)
    tpreds = model.predict(X_test_cv)
    print(accuracy_score(y_test_cv, tpreds))
