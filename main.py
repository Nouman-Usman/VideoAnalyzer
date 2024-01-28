import os
import numpy as np
import cv2
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras import layers, models
from webencodings import labels

num_frames_per_video = 10
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')
video_path = "C:\\Users\\Nouman\\Downloads\\Video"


def extract_frames(video_path, num_frames=num_frames_per_video):
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(num_frames):
        frame_idx = int((i / num_frames) * total_frames)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        frames.append(frame)

    cap.release()
    return np.array(frames)


def preprocess_frames(frames):
    frames = frames / 255.0
    return frames


video_frames_data = []
for video_file in os.listdir(video_path):
    video_file_path = os.path.join(video_path, video_file)
    frames = extract_frames(video_file_path)
    frames = preprocess_frames(frames)
    video_frames_data.append(frames)

text_data = ["Your video transcript goes here.", "Another video transcript."]
tokenized_input = tokenizer(text_data, padding=True, truncation=True, return_tensors="tf")
bert_output_text = bert_model(**tokenized_input)
video_data_processed = np.stack(video_frames_data)


def create_and_train_model(tokenized_input, video_data_processed, labels):
    # CNN model for video frame processing
    image_model = models.Sequential([
        layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu'), input_shape=(None, 224, 224, 3)),
        layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
        layers.TimeDistributed(layers.Flatten()),
        layers.LSTM(64, return_sequences=True),
    ])

    video_output = image_model(video_data_processed)
    bert_output_text = bert_model(**tokenized_input)
    combined_output = layers.concatenate([bert_output_text['last_hidden_state'][:, 0, :], video_output[:, -1, :]])
    dense_layer = layers.Dense(128, activation='relu')(combined_output)
    output_layer = layers.Dense(4, activation='softmax')(dense_layer)  # Assuming num_classes is defined
    model = models.Model(inputs=[bert_model.input, image_model.input], outputs=output_layer)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit([tokenized_input, video_data_processed], labels, epochs=5, batch_size=32)

    return model


trained_model = create_and_train_model(tokenized_input, video_data_processed, 2)
