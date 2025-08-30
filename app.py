from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
import numpy as np
from PIL import Image
import io
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, GPT2TokenizerFast
import pickle

app = Flask(__name__)
CORS(app)  # Enable CORS to allow frontend requests

# Load the custom ResNet50-based model
custom_model = tf.keras.models.load_model('image_caption_generator_resnet50_epochs8.keras')

# Load the pre-saved tokenizer from pickle file
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

# Get vocab size from loaded tokenizer
vocab_size = len(tokenizer.word_index) + 1

# Define max_length
max_length = 35

# Load ResNet50 model for feature extraction
resnet50 = ResNet50()
resnet50 = Model(inputs=resnet50.inputs, outputs=resnet50.layers[-2].output)

# Load ViT+GPT2 model
vit_gpt2_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
vit_tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

def preprocess_image_custom(image):
    # Resize image to 224x224 (standard for ResNet50)
    image = image.resize((224, 224))
    # Convert to array and preprocess for ResNet50
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)
    return image_array

def preprocess_image_vit(image):
    # Process image for ViT
    img = image_processor(image, return_tensors="pt")
    return img

def extract_features(image_array):
    # Extract ResNet50 features
    features = resnet50.predict(image_array, verbose=0)
    return features

def index_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_caption_custom(model, image_features, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)[0]
        sequence = sequence.reshape((1, sequence.shape[0]))
        yhat = model.predict([image_features, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = index_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break
    return in_text.replace('startseq', '').replace('endseq', '').strip()

def generate_caption_vit(model, image_processor, tokenizer, image):
    img = preprocess_image_vit(image)
    output = model.generate(**img)
    caption = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    return caption

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/generate_caption', methods=['POST'])
def generate_caption_endpoint():
    if 'image' not in request.files or 'model' not in request.form:
        return jsonify({'error': 'Image and model selection required'}), 400

    file = request.files['image']
    model_choice = request.form['model']

    try:
        # Open image
        image = Image.open(io.BytesIO(file.read())).convert('RGB')

        if model_choice == 'custom':
            # Process for custom ResNet50 model
            image_array = preprocess_image_custom(image)
            image_features = extract_features(image_array)
            caption = generate_caption_custom(custom_model, image_features, tokenizer, max_length)
        elif model_choice == 'transformer':
            # Process for ViT+GPT2 model
            caption = generate_caption_vit(vit_gpt2_model, image_processor, vit_tokenizer, image)
        else:
            return jsonify({'error': 'Invalid model selection'}), 400

        return jsonify({'caption': caption})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)