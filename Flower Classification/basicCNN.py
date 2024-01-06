import keras
import pandas as pd
import os
import numpy as np

import tensorflow as tf
import matplotlib as plt

from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score, ConfusionMatrixDisplay


path = 'dataset'
categories = os.listdir(path)

print(categories)

X, y = [], []
# read data
for category in categories:
    folderpath = path + '/' + category
    for img in os.listdir(folderpath):
        img_path = folderpath + '/' + img
        if img_path.endswith('.jpg') or img_path.endswith('.jpeg'):
            X.append(img_path)
            y.append(category)



# split data to train, test, valid

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=101)

train_df = pd.DataFrame({'FileName':X_train, 'Category':y_train})
test_df = pd.DataFrame({'FileName':X_test, 'Category':y_test})
val_df = pd.DataFrame({'FileName':X_val, 'Category':y_val})

# transform img
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                  horizontal_flip=True,
                                  shear_range=0.2,
                                  zoom_range=0.2)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_dataframe(train_df,
                                                x_col='FileName',
                                                y_col='Category',
                                                batch_size=32,
                                                class_mode='categorical',
                                                target_size=(64, 64))

test_set = train_datagen.flow_from_dataframe(test_df,
                                                x_col='FileName',
                                                y_col='Category',
                                                batch_size=32,
                                                class_mode='categorical',
                                                target_size=(64, 64))

validation_set = train_datagen.flow_from_dataframe(val_df,
                                                x_col='FileName',
                                                y_col='Category',
                                                batch_size=32,
                                                class_mode='categorical',
                                                target_size=(64, 64))

# model train

train_model = tf.keras.models.Sequential()
# first
train_model.add(tf.keras.layers.Conv2D(filters=32,
                              kernel_size=3,
                              activation='relu',
                              input_shape=[64,64,3]))

train_model.add(tf.keras.layers.MaxPool2D(pool_size=2,
                                 strides=2))

#second convolution layer
train_model.add(tf.keras.layers.Conv2D(filters=32,
                              kernel_size=3,
                              activation='relu'))
train_model.add(tf.keras.layers.MaxPool2D(pool_size=2,
                                 strides=2))

#flattern
train_model.add(tf.keras.layers.Flatten())

#full connection
train_model.add(tf.keras.layers.Dense(units=128, activation='relu'))
#output layer
train_model.add(tf.keras.layers.Dense(units=len(categories), activation='sigmoid'))

train_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


history1 = train_model.fit(x=training_set, validation_data=validation_set, epochs=100)

train_model.summary()

loss, accuracy= train_model.evaluate(validation_set)
print(f"Loss is:{loss}")
print(f"Accuracy is:{accuracy}")
history= pd.DataFrame(history1.history)
history.head()

plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.plot(history['loss'],label='Train_Loss')
plt.plot(history['val_loss'],label='Validation_Loss')
plt.title('Train_Loss and Validation_Loss',fontsize=15)
plt.legend()


plt.subplot(1,2,2)
plt.plot(history['accuracy'],label='Train_Accuracy')
plt.plot(history['val_accuracy'],label='Validation_Accuracy')
plt.title('Train_Accuracy and Validation_Accuracy',fontsize=15)
plt.legend()
plt.show()


loss, accuracy= train_model.evaluate(test_set)
print(f"Loss is:{loss}")
print(f"Accuracy is:{accuracy}")

x_val=[]
y_val=[]
y_pred=[]

for images, labels in test_set:
    x_val.extend(images)
    y_val.extend(labels)
predictions=train_model.predict(np.array(x_val))
for i in predictions:
    y_pred.append(np.argmax(i))
plt.figure(figsize=(50, 50),tight_layout=True)

for i in range(32):
    ax = plt.subplot(8, 4, i + 1)
    plt.imshow(x_val[i].astype("uint8"))
    actual_label = categories[y_val[i]]
    predicted_label = categories[y_pred[i]]

    # Check if the actual and predicted labels are the same
    if actual_label == predicted_label:
        label_color = 'green'
    else:
        label_color = 'red'

    plt.title(f'Actual: {actual_label} \n Predicted: {predicted_label}', color=label_color, fontsize=35)
    plt.axis("off")

plt.show()


f1_scores=[]
precision_scores=[]
recall_scores=[]

f1_scores.append(f1_score(y_pred, y_val, average="macro"))
precision_scores.append(precision_score(y_pred, y_val, average="macro"))
recall_scores.append(recall_score(y_pred, y_val, average="macro"))


# Print the Results
print(f"F1-Score:{f1_scores}")
print(f"Precision:{precision_scores}")
print(f"Recall:{recall_scores}")


print("Classification_Report")
print("-----------------------")
print(classification_report(y_val,y_pred))
print("Confusion_Matrix")
print("----------------------")
ConfusionMatrixDisplay.from_predictions(y_val, y_pred, display_labels=categories, xticks_rotation="vertical")
plt.show()

model_name=["Basic CNN"]
result_df= pd.DataFrame({"F1_Score":f1_scores, "Precision_Score":precision_scores, "Recall_Score":recall_scores}, index=model_name)
result_df= result_df.T.sort_values(by="ResNet50", ascending=False)
print(result_df)

result_df.plot(kind="bar",figsize=(5,5), color="Grey").legend(bbox_to_anchor=(1.5,1))
