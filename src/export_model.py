#exporting  model  for deployment.

from model_custom import build_model
import tensorflow as tf

model = build_model()
model.save('complete_dog_cat_model.keras')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('dog_cat_model.tflite', 'wb') as f:
    f.write(tflite_model)
