
from src.backbone import TFLiteModel, get_model
import tensorflow as tf


model = get_model()

# Load weights from the weights file.
model.load_weights('models\islr-fp16-192-8-seed42-fold0-best.h5')

tflite_keras_model = TFLiteModel(islr_models=[model])

# Get the concrete function from the tf.function
concrete_function = model.__call__.get_concrete_function(tf.TensorSpec(shape=[None, 543, 3], dtype=tf.float32))

# Convert the model to TFLite format
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_function])
tflite_model = converter.convert()

# Save the TFLite model to a file
with open('tflite_model.tflite', 'wb') as f:
    f.write(tflite_model)