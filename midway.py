import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("doctorg_model.h5")
tflite_model = converter.convert()

with open("doctorg_model.tflite", "wb") as f:
   f.write(tflite_model)