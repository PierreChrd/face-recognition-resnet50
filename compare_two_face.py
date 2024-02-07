from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf

def euclidean_distance(embedding1, embedding2):
    """Calculate the Euclidean distance between two embeddings."""
    return np.linalg.norm(embedding1 - embedding2)

# Triplet loss function as defined during training
def triplet_loss(y_true, y_pred, alpha = 0.4):
    total_length = y_pred.shape.as_list()[-1]
    anchor, positive, negative = y_pred[:,:int(1/3*total_length)], y_pred[:,int(1/3*total_length):int(2/3*total_length)], y_pred[:,int(2/3*total_length):]

    pos_dist = tf.reduce_sum(tf.square(anchor-positive), axis=1)
    neg_dist = tf.reduce_sum(tf.square(anchor-negative), axis=1)

    basic_loss = pos_dist - neg_dist + alpha
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0))
    return loss

# Load the saved model
model = load_model('my_resnet50_triplet_model.h5', custom_objects={'triplet_loss': triplet_loss})
print("Model loaded successfully.")

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(64, 64))  # Update size if different
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# Path to your image
image_path = 'photos/output_fae09905-20ae-4559-bad6-f78f0dd9b067.jpg'

# Preprocess the image
preprocessed_img = preprocess_image(image_path)

embeddings = model.predict([preprocessed_img, preprocessed_img, preprocessed_img])

# Extract the embeddings (the first part of the output)
embedding_output = embeddings

print("Embedding for the image:")
print(embedding_output)

# Path to your image
image_path = 'photos/output_7eca5148-e9ef-4e0d-b855-2ed196ed4ba9.jpg'

# Preprocess the image
preprocessed_img = preprocess_image(image_path)

embeddings = model.predict([preprocessed_img, preprocessed_img, preprocessed_img])

# Extract the embeddings
embedding_output_2 = embeddings

print("Embedding for the image:")
print(embedding_output)

distance = euclidean_distance(embedding_output, embedding_output_2)
print("Euclidean Distance:", distance)