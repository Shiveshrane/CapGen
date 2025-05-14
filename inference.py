import os
import re
import numpy as np
import pandas as pd
import json
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import efficientnet
from tensorflow.keras.layers import TextVectorization

# Import the model from model.py
from model import caption_model, IMAGE_SIZE, VOCAB_SIZE, SEQ_LENGTH, EMBED_DIM, FF_DIM

BATCH_SIZE = 32
EPOCHS = 30
AUTOTUNE = tf.data.AUTOTUNE

# Define strip_chars for text standardization
strip_chars = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
strip_chars = strip_chars.replace("<", "")
strip_chars = strip_chars.replace(">", "")

# Load vocabulary
with open("vocab.json", "r") as f:
    json_vocab = json.load(f)

def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")

# Create vectorization layer
vectorization = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=SEQ_LENGTH,
    standardize=custom_standardization,
    vocabulary=json_vocab
)

# Define image preprocessing function
def decode_and_resize(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

vocab = json_vocab
index_lookup = dict(zip(range(len(vocab)), vocab))
max_decoded_sentence_length = SEQ_LENGTH - 1


def generate_caption(image_path):
    """Generate caption for an input image"""
    # Read the image from the disk
    try:
        sample_img = decode_and_resize(image_path)
        
        # Pass the image to the CNN
        img = tf.expand_dims(sample_img, 0)
        img = caption_model.cnn_model(img)

        # Pass the image features to the Transformer encoder
        encoded_img = caption_model.encoder(img, training=False)

        # Generate the caption using the Transformer decoder
        decoded_caption = "<start> "
        for i in range(max_decoded_sentence_length):
            tokenized_caption = vectorization([decoded_caption])[:, :-1]
            mask = tf.math.not_equal(tokenized_caption, 0)
            predictions = caption_model.decoder(
                tokenized_caption, encoded_img, training=False, mask=mask
            )
            sampled_token_index = np.argmax(predictions[0, i, :])
            sampled_token = index_lookup[sampled_token_index]
            if sampled_token == " <end>":
                break
            decoded_caption += " " + sampled_token

        decoded_caption = decoded_caption.replace("<start> ", "")
        decoded_caption = decoded_caption.replace(" <end>", "").strip()
        
        return decoded_caption, sample_img.numpy()
    except Exception as e:
        return str(e), None

# Streamlit App
def main():
    st.title("Image Caption Generator")
    st.write("Upload an image and get its caption!")
    
    # Load model weights
    try:
        caption_model.load_weights("capgen_weights.h5")
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model weights: {str(e)}")
        st.info("Please make sure 'capgen_weights.h5' is in the correct directory.")
        return
        
    # File uploader for images
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())
          # Display the image
        image = Image.open("temp_image.jpg")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Generate caption when button is pressed
        if st.button("Generate Caption"):
            with st.spinner("Generating caption..."):
                caption, _ = generate_caption("temp_image.jpg")
            
            st.success("Caption Generated!")
            st.write(f"**Caption:** {caption}")

if __name__ == "__main__":
    main()
