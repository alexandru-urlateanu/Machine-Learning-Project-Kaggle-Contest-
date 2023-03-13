# Machine-Learning-Project-Kaggle-Contest

Translation Source Dialect Identification

The goal of the competition is to predict the native-dialect of a text based on its translation in different languages.

__Dataset Description__

The training data consists of a csv file with 41570 training examples containing the translation of the original content in the three native dialects (English, Scottish, Irish) to five languages: Danish, German, Spanish, Italian, Dutch. Each text in the native dialect is translated to the five languages, so for each language there are 8314 training examples, a fifth of the total 41570 training examples. The csv file contains three columns: language, text and label. The test set consists of another csv file with 18360 test examples. The language and test labels are not provided with the data. Only the text of the translated original content is provided.

__File descriptions__
* train_data.csv - the training set containing the training examples (language, text) and labels
* test_data.csv - the test set containing the test examples (only text) (without labels)
* sampleSubmission.csv - a sample submission file in the correct format

