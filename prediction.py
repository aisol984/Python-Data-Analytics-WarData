# Random Forest Regression for Predition 3rd War

# Import the libraries for data preprocessing
import numpy as np
import pandas as pd
import re
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

# Analyze Data
def get_tokenizer():
    # Importing the dataset
    dataset = pd.read_excel('Data3.xlsx')
    for i in range(2, 5):
        # Make all letters to lower
        dataset.iloc[:, i] = dataset.iloc[:, i].apply(lambda x:x.lower())
        # Remove Space of all words
        dataset.iloc[:, i] = dataset.iloc[:, i].apply(lambda x:re.sub('[^a-zA-Z0-9,\s]', "", x))
        dataset.iloc[:, i] = dataset.iloc[:, i].apply(lambda x:x.replace(' ', ''))
        # Split words by comma
        dataset.iloc[:, i] = dataset.iloc[:, i].apply(lambda x:x.split(','))
    
    # Convert all words in `Location`, `Countires Involved` columns to integer
    tokenizer = Tokenizer(num_words=300, split=",")
    tokenizer.fit_on_texts(dataset.iloc[:, 3])
    # Convert all words in `Reason` columns to integer
    tokenizer_reason = Tokenizer(num_words=15, split=",")
    tokenizer_reason.fit_on_texts(dataset.iloc[:, 4])
    return tokenizer, tokenizer_reason

# Preprocessing all output columns
def preprocessing_output_data(input, dataframe):
    dataset = dataframe.copy()
    tokenizer, tokenizer_reason = get_tokenizer()
    # Make all letters to lower
    dataset.iloc[:, 5] = dataset.iloc[:, 5].apply(lambda x:x.lower())
    # Remove Space of all words
    dataset.iloc[:, 5] = dataset.iloc[:, 5].apply(lambda x:re.sub('[^a-zA-Z0-9,\s]', "", x))
    dataset.iloc[:, 5] = dataset.iloc[:, 5].apply(lambda x:x.replace(' ', ''))
    # Split words by comma
    dataset.iloc[:, 5] = dataset.iloc[:, 5].apply(lambda x:x.split(','))

    dataset.iloc[:, 5] = tokenizer.texts_to_sequences(dataset.iloc[:, 5])
    dataset["Winner"] = pad_sequences(dataset.iloc[:, 5]).tolist()
    y = dataset.iloc[:, 5].values
    y_temp = []
    for i in range(len(y)):
        temp = []
        for j in range(len(y[i])):
            if y[i][j] == 0:
                temp.append(y[i][j])
                continue
            for k in range(3, 9):
                if input[i][k] == y[i][j]:
                    y[i][j] = k
                    break
            temp.append(y[i][j])
        y_temp.append(temp)
    y = y_temp
    return y

# Preprocessing all input columns
def preprocessing_input_data(dataframe):
    dataset = dataframe.copy()
    tokenizer, tokenizer_reason = get_tokenizer()
    for i in range(2, 5):
        # Make all letters to lower
        dataset.iloc[:, i] = dataset.iloc[:, i].apply(lambda x:x.lower())
        # Remove Space of all words
        dataset.iloc[:, i] = dataset.iloc[:, i].apply(lambda x:re.sub('[^a-zA-Z0-9,\s]', "", x))
        dataset.iloc[:, i] = dataset.iloc[:, i].apply(lambda x:x.replace(' ', ''))
        # Split words by comma
        dataset.iloc[:, i] = dataset.iloc[:, i].apply(lambda x:x.split(','))
    
    for i in range(2, 5):
        if i == 4:
            dataset.iloc[:, 4] = tokenizer_reason.texts_to_sequences(dataset.iloc[:, 4])
            continue
        dataset.iloc[:, i] = tokenizer.texts_to_sequences(dataset.iloc[:, i])

    # Make all rows into same length vector
    dataset["Location"] = pad_sequences(dataset.iloc[:, 2], maxlen=3).tolist()
    dataset["Countries Involved"] = pad_sequences(dataset.iloc[:, 3], maxlen=6).tolist()
    dataset["Reasons"] = pad_sequences(dataset.iloc[:, 4]).tolist()
    x = dataset.iloc[:, 2:5].values
    x_temp = []
    for i in range(len(x)):
        temp = []
        for j in range(len(x[i])):
            for k in range(len(x[i][j])):
                temp.append(x[i][j][k])
        x_temp.append(temp)
    x = x_temp
    return x

dataset = pd.read_excel('Data3.xlsx')
x = preprocessing_input_data(dataset)
y = preprocessing_output_data(x, dataset)

# Split dataset as training data and test data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Fitting K Mean Classification to the Training Set
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
classifier = MultiOutputClassifier(KNeighborsClassifier(n_neighbors=10, metric='minkowski', p =2))
classifier.fit(x_train, y_train)

# Evaluation Model
y_pred = classifier.predict(x_test)
count = 0
for i in range(len(y_pred)):
    if y_pred[i][1] == y_test[i][1]:
        count = count + 1


# Test model
date = '2010'
name = 'World War III'
location = 'China'
countries = 'China, Ghana, UK'
countries_list = countries.split(",")
reason = 'Revolution'
df_test = pd.DataFrame(np.array([[date, name, location, countries, reason]]), columns=['Date', 'Name', 'Location', 'Countries Involved', 'Reason'])
test_input = preprocessing_input_data(df_test)
test_pred = classifier.predict(test_input)
for i in range(2):
    if test_pred[0][i] == 0:
        continue
    print(countries_list[test_pred[0][i] - 9 + len(countries_list)])