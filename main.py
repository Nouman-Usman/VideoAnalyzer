# import os
# import numpy as np
# import cv2
# import tensorflow as tf
# from transformers import BertTokenizer, TFBertModel
# from tensorflow.keras import layers, models
# from webencodings import labels
#
# num_frames_per_video = 10
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# bert_model = TFBertModel.from_pretrained('bert-base-uncased')
# video_path = "C:\\Users\\Nouman\\Downloads\\Video"
#
#
# def extract_frames(video_path, num_frames=num_frames_per_video):
#     frames = []
#     cap = cv2.VideoCapture(video_path)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#
#     for i in range(num_frames):
#         frame_idx = int((i / num_frames) * total_frames)
#         cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame = cv2.resize(frame, (224, 224))
#         frames.append(frame)
#
#     cap.release()
#     return np.array(frames)
#
#
# def preprocess_frames(frames):
#     frames = frames / 255.0
#     return frames
#
#
# video_frames_data = []
# for video_file in os.listdir(video_path):
#     video_file_path = os.path.join(video_path, video_file)
#     frames = extract_frames(video_file_path)
#     frames = preprocess_frames(frames)
#     video_frames_data.append(frames)
#
# text_data = ["Your video transcript goes here.", "Another video transcript."]
# tokenized_input = tokenizer(text_data, padding=True, truncation=True, return_tensors="tf")
# bert_output_text = bert_model(**tokenized_input)
# video_data_processed = np.stack(video_frames_data)
#
#
# def create_and_train_model(tokenized_input, video_data_processed, labels):
#     # CNN model for video frame processing
#     image_model = models.Sequential([
#         layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu'), input_shape=(None, 224, 224, 3)),
#         layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
#         layers.TimeDistributed(layers.Flatten()),
#         layers.LSTM(64, return_sequences=True),
#     ])
#
#     video_output = image_model(video_data_processed)
#     bert_output_text = bert_model(**tokenized_input)
#     combined_output = layers.concatenate([bert_output_text['last_hidden_state'][:, 0, :], video_output[:, -1, :]])
#     dense_layer = layers.Dense(128, activation='relu')(combined_output)
#     output_layer = layers.Dense(4, activation='softmax')(dense_layer)  # Assuming num_classes is defined
#     model = models.Model(inputs=[bert_model.input, image_model.input], outputs=output_layer)
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     model.fit([tokenized_input, video_data_processed], labels, epochs=5, batch_size=32)
#
#     return model
#
#
# trained_model = create_and_train_model(tokenized_input, video_data_processed, 2)
# AIzaSyCZAM5a2K0eHOZuymvXRxo8c2XyJxBumiw

import requests
import geocoder


def get_current_location():
    # Get the current location using the IP address
    location = geocoder.ip('me')

    if location.ok:
        latitude, longitude = location.latlng
        return f"{latitude},{longitude}"
    else:
        print("Unable to retrieve current location.")
        return None
def get_nearest_shop(api_key, shop_type, location):
    base_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

    params = {
        'key': api_key,
        'location': location,  # Latitude and longitude in the format "lat,lng"
        'radius': 10000,  # You can adjust the radius based on your preferences
        'types': shop_type
    }

    response = requests.get(base_url, params=params)
    results = response.json().get('results', [])

    if results:
        nearest_shop = results[0]
        name = nearest_shop.get('name', 'N/A')
        address = nearest_shop.get('vicinity', 'N/A')
        place_id = nearest_shop.get('place_id', 'N/A')

        print(f"Nearest {shop_type} shop: {name}")
        print(f"Address: {address}")
        print(f"Place ID: {place_id}")

        # You can also construct the Google Maps URL for navigation
        maps_url = f"https://www.google.com/maps/place/?q=place_id:{place_id}"
        print(f"Google Maps URL: {maps_url}")
    else:
        print(f"No {shop_type} shops found nearby.")

# Example usage
api_key = "AIzaSyCZAM5a2K0eHOZuymvXRxo8c2XyJxBumiw"
current_location = get_current_location()
shop_type = "computer_repair"
location = "37.7749,-122.4194"
get_nearest_shop(api_key, shop_type, current_location)

