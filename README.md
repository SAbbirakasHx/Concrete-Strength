🧱 Concrete Crack Image Detection

Concrete Crack Image Detection
Given images of concrete surfaces, let's try to detect cracks in the concrete.

We will use a TensorFlow CNN to make our predictions.

Getting Started
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from pathlib import Path
from sklearn.model_selection import train_test_split

import tensorflow as tf

from sklearn.metrics import confusion_matrix, classification_report
positive_dir = Path('../input/surface-crack-detection/Positive')
negative_dir = Path('../input/surface-crack-detection/Negative')
Creating DataFrames
def generate_df(image_dir, label):
    filepaths = pd.Series(list(image_dir.glob(r'*.jpg')), name='Filepath').astype(str)
    labels = pd.Series(label, name='Label', index=filepaths.index)
    df = pd.concat([filepaths, labels], axis=1)
    return df
positive_df = generate_df(positive_dir, label="POSITIVE")
negative_df = generate_df(negative_dir, label="NEGATIVE")

all_df = pd.concat([positive_df, negative_df], axis=0).sample(frac=1.0, random_state=1).reset_index(drop=True)
all_df
Filepath	Label
0	../input/surface-crack-detection/Positive/0574...	POSITIVE
1	../input/surface-crack-detection/Positive/1870...	POSITIVE
2	../input/surface-crack-detection/Positive/0967...	POSITIVE
3	../input/surface-crack-detection/Negative/0791...	NEGATIVE
4	../input/surface-crack-detection/Positive/1400...	POSITIVE
...	...	...
39995	../input/surface-crack-detection/Positive/0854...	POSITIVE
39996	../input/surface-crack-detection/Negative/1944...	NEGATIVE
39997	../input/surface-crack-detection/Positive/0977...	POSITIVE
39998	../input/surface-crack-detection/Positive/1504...	POSITIVE
39999	../input/surface-crack-detection/Negative/1099...	NEGATIVE
40000 rows × 2 columns

train_df, test_df = train_test_split(
    all_df.sample(6000, random_state=1),
    train_size=0.7,
    shuffle=True,
    random_state=1
)
Loading Image Data
train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

test_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)
train_data = train_gen.flow_from_dataframe(
    train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(120, 120),
    color_mode='rgb',
    class_mode='binary',
    batch_size=32,
    shuffle=True,
    seed=42,
    subset='training'
)

val_data = train_gen.flow_from_dataframe(
    train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(120, 120),
    color_mode='rgb',
    class_mode='binary',
    batch_size=32,
    shuffle=True,
    seed=42,
    subset='validation'
)

test_data = train_gen.flow_from_dataframe(
    test_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(120, 120),
    color_mode='rgb',
    class_mode='binary',
    batch_size=32,
    shuffle=False,
    seed=42
)
Found 3360 validated image filenames belonging to 2 classes.
Found 840 validated image filenames belonging to 2 classes.
Found 1800 validated image filenames belonging to 2 classes.
Training
inputs = tf.keras.Input(shape=(120, 120, 3))
x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(inputs)
x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)
x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print(model.summary())
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 120, 120, 3)]     0         
_________________________________________________________________
conv2d (Conv2D)              (None, 118, 118, 16)      448       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 59, 59, 16)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 57, 57, 32)        4640      
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 28, 28, 32)        0         
_________________________________________________________________
global_average_pooling2d (Gl (None, 32)                0         
_________________________________________________________________
dense (Dense)                (None, 1)                 33        
=================================================================
Total params: 5,121
Trainable params: 5,121
Non-trainable params: 0
_________________________________________________________________
None
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=100,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
    ]
)
Epoch 1/100
105/105 [==============================] - 44s 386ms/step - loss: 0.6857 - accuracy: 0.5471 - val_loss: 0.6396 - val_accuracy: 0.7857
Epoch 2/100
105/105 [==============================] - 8s 76ms/step - loss: 0.6254 - accuracy: 0.7250 - val_loss: 0.5556 - val_accuracy: 0.8048
Epoch 3/100
105/105 [==============================] - 8s 81ms/step - loss: 0.5395 - accuracy: 0.8011 - val_loss: 0.4688 - val_accuracy: 0.8738
Epoch 4/100
105/105 [==============================] - 8s 78ms/step - loss: 0.4359 - accuracy: 0.8723 - val_loss: 0.3775 - val_accuracy: 0.9179
Epoch 5/100
105/105 [==============================] - 8s 77ms/step - loss: 0.3400 - accuracy: 0.9243 - val_loss: 0.2847 - val_accuracy: 0.9262
Epoch 6/100
105/105 [==============================] - 8s 77ms/step - loss: 0.2753 - accuracy: 0.9352 - val_loss: 0.2371 - val_accuracy: 0.9512
Epoch 7/100
105/105 [==============================] - 9s 83ms/step - loss: 0.2112 - accuracy: 0.9497 - val_loss: 0.2007 - val_accuracy: 0.9631
Epoch 8/100
105/105 [==============================] - 8s 77ms/step - loss: 0.1713 - accuracy: 0.9553 - val_loss: 0.1990 - val_accuracy: 0.9548
Epoch 9/100
105/105 [==============================] - 8s 74ms/step - loss: 0.1562 - accuracy: 0.9571 - val_loss: 0.1559 - val_accuracy: 0.9667
Epoch 10/100
105/105 [==============================] - 8s 78ms/step - loss: 0.1416 - accuracy: 0.9563 - val_loss: 0.1698 - val_accuracy: 0.9690
Epoch 11/100
105/105 [==============================] - 9s 83ms/step - loss: 0.1327 - accuracy: 0.9648 - val_loss: 0.1375 - val_accuracy: 0.9714
Epoch 12/100
105/105 [==============================] - 8s 78ms/step - loss: 0.1213 - accuracy: 0.9634 - val_loss: 0.1309 - val_accuracy: 0.9750
Epoch 13/100
105/105 [==============================] - 8s 76ms/step - loss: 0.0997 - accuracy: 0.9686 - val_loss: 0.1269 - val_accuracy: 0.9750
Epoch 14/100
105/105 [==============================] - 8s 78ms/step - loss: 0.0988 - accuracy: 0.9716 - val_loss: 0.1204 - val_accuracy: 0.9786
Epoch 15/100
105/105 [==============================] - 9s 84ms/step - loss: 0.0875 - accuracy: 0.9741 - val_loss: 0.1191 - val_accuracy: 0.9786
Epoch 16/100
105/105 [==============================] - 8s 76ms/step - loss: 0.0903 - accuracy: 0.9704 - val_loss: 0.1160 - val_accuracy: 0.9786
Epoch 17/100
105/105 [==============================] - 8s 73ms/step - loss: 0.0831 - accuracy: 0.9756 - val_loss: 0.1293 - val_accuracy: 0.9774
Epoch 18/100
105/105 [==============================] - 8s 75ms/step - loss: 0.0727 - accuracy: 0.9769 - val_loss: 0.1132 - val_accuracy: 0.9750
Epoch 19/100
105/105 [==============================] - 9s 83ms/step - loss: 0.0716 - accuracy: 0.9761 - val_loss: 0.1101 - val_accuracy: 0.9821
Epoch 20/100
105/105 [==============================] - 8s 73ms/step - loss: 0.0843 - accuracy: 0.9712 - val_loss: 0.1078 - val_accuracy: 0.9762
Epoch 21/100
105/105 [==============================] - 8s 77ms/step - loss: 0.0757 - accuracy: 0.9765 - val_loss: 0.1081 - val_accuracy: 0.9774
Epoch 22/100
105/105 [==============================] - 8s 81ms/step - loss: 0.0602 - accuracy: 0.9766 - val_loss: 0.1202 - val_accuracy: 0.9821
Epoch 23/100
105/105 [==============================] - 8s 81ms/step - loss: 0.0644 - accuracy: 0.9795 - val_loss: 0.1065 - val_accuracy: 0.9810
Epoch 24/100
105/105 [==============================] - 8s 78ms/step - loss: 0.0629 - accuracy: 0.9787 - val_loss: 0.1644 - val_accuracy: 0.9738
Epoch 25/100
105/105 [==============================] - 8s 74ms/step - loss: 0.0671 - accuracy: 0.9772 - val_loss: 0.1070 - val_accuracy: 0.9821
Epoch 26/100
105/105 [==============================] - 9s 84ms/step - loss: 0.0544 - accuracy: 0.9794 - val_loss: 0.1078 - val_accuracy: 0.9821
fig = px.line(
    history.history,
    y=['loss', 'val_loss'],
    labels={'index': "Epoch", 'value': "Loss"},
    title="Training and Validation Loss Over Time"
)

fig.show()
Results
def evaluate_model(model, test_data):
    
    results = model.evaluate(test_data, verbose=0)
    loss = results[0]
    acc = results[1]
    
    print("    Test Loss: {:.5f}".format(loss))
    print("Test Accuracy: {:.2f}%".format(acc * 100))
    
    y_pred = np.squeeze((model.predict(test_data) >= 0.5).astype(np.int))
    cm = confusion_matrix(test_data.labels, y_pred)
    clr = classification_report(test_data.labels, y_pred, target_names=["NEGATIVE", "POSITIVE"])
    
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='g', vmin=0, cmap='Blues', cbar=False)
    plt.xticks(ticks=np.arange(2) + 0.5, labels=["NEGATIVE", "POSITIVE"])
    plt.yticks(ticks=np.arange(2) + 0.5, labels=["NEGATIVE", "POSITIVE"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
    
    print("Classification Report:\n----------------------\n", clr)
evaluate_model(model, test_data)
    Test Loss: 0.10731
Test Accuracy: 97.17%
