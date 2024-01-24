import cv2
import os
import numpy as np
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras import layers, models

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

# Define constants
video_path = "C:\\Users\\Nouman\\Downloads\\Video"
text_data = ["Your video transcript goes here.", "Another video transcript."]
num_frames_per_video = 10  # Adjust as needed

# Function to extract frames from a video
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

        # Resize frame to match your model's input size (e.g., 224x224)
        frame = cv2.resize(frame, (224, 224))
        frames.append(frame)

    cap.release()
    return np.array(frames)

# Function to preprocess video frames
def preprocess_frames(frames):
    # Normalize pixel values to be between 0 and 1
    frames = frames / 255.0
    return frames

# Preprocess video frames for each video
video_frames_data = []
for video_file in os.listdir(video_path):
    video_file_path = os.path.join(video_path, video_file)
    frames = extract_frames(video_file_path)
    frames = preprocess_frames(frames)
    video_frames_data.append(frames)

# Tokenize and encode text data using BERT tokenizer
tokenized_input = tokenizer(text_data, padding=True, truncation=True, return_tensors="tf")
bert_output = bert_model(**tokenized_input)

# Stack video frames for input to the model
video_data_processed = np.stack(video_frames_data)

# Create the final model (as shown in the previous example)
# ...

# Train the model with your data (as shown in the previous example)
# ...

# Use the model for predictions (as shown in the previous example)
# ...