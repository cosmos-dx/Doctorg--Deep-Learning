import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformerModelCardData

df = pd.read_csv('doctorg_processed.csv')

label_encoder = LabelEncoder()
df["disease_encoded"] = label_encoder.fit_transform(df["name"])

X = np.vstack(df.apply(lambda row: np.fromstring(row["embeddings"].strip("[]"), sep=' ') * row["weight"], axis=1).values)
y = tf.keras.utils.to_categorical(df["disease_encoded"], num_classes=len(label_encoder.classes_))

model = Sequential([
    Dense(128, activation="relu", input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(len(label_encoder.classes_), activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X, y, epochs=50, batch_size=16, validation_split=0.2)

model.export("doctorg_model.h5")
# model.save("disease_prediction_model_local.h5")


