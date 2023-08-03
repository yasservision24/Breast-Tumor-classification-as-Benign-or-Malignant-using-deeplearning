import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split
import tensorflow as tf

dataset = sklearn.datasets.load_breast_cancer()
print(dataset)
features_data = pd.DataFrame(dataset.data, columns=dataset.feature_names)
labels_data=dataset.target
print(labels_data)

unique, counts = np.unique(labels_data, return_counts=True)
combined_data = np.c_[dataset.data, dataset.target]

features_n_labels = list(dataset.feature_names) + ['target']
data_f = pd.DataFrame(combined_data, columns=features_n_labels)
display(data_f.head(15))
display(data_f.tail(15))
data_f.info()

data_f.isnull().sum()
data_f.isnull().sum().sum()
data_f['target'].value_counts()

X_train, X_test, Y_train, Y_test = train_test_split(features_data,labels_data, test_size=0.2, random_state=2)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

from tensorflow import keras
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)


import tensorflow as tf
tf.random.set_seed(3)
from tensorflow import keras



model = keras.Sequential([
                          keras.layers.Flatten(input_shape=(30,)),
                          keras.layers.Dense(20, activation='relu'),
                          keras.layers.Dense(2, activation='sigmoid')
])



model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(X_train_std, Y_train, validation_split=0.1, epochs=100)



plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training data', 'validation data'], loc = 'lower right')

loss, accuracy = model.evaluate(X_test_std, Y_test)
print(accuracy)

print(X_test_std.shape)
print(X_test_std[0])

# (probability of being Bening,probability of being Malignant)
l=[0.25,0.56]
max=np.argmax(l)
print(l)
print(max)





3 HERE WE ARE USING THE MODEL FOR PREDICTION

# here we are providing the features to the model to get the output label
# imp point to keep in mind, as we are providing the input as array it is must be in sequnce with the features of dataframe 


input = (11.76,21.6,74.72,427.9,0.08637,0.04966,0.01657,0.01115,0.1495,0.05888,0.4062,1.21,2.635,28.47,0.005857,0.009758,0.01168,0.007445,0.02406,0.001769,12.98,25.72,82.98,516.5,0.1085,0.08615,0.05523,0.03715,0.2433,0.06563)

input_array = np.asarray(input)

input_data_reshaped = input_array.reshape(1,-1)
input_data_std = scaler.transform(input_data_reshaped)
prediction = model.predict(input_data_std)
print(prediction)
prediction_label = [np.argmax(prediction)]
print(prediction_label)
if(prediction_label[0] == 0):
  print('The tumor is Malignant')
else:
  print('The tumor is Benign')
