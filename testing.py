import cv2
import os
import numpy as np
from tensorflow.keras import layers, models
from transformers import BertTokenizer, TFBertModel

# Define constants
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')
video_path = "C:\\Users\\Nouman\\Downloads\\Video"
num_frames_per_video = 10  # Adjust as needed
num_classes = 2  # Replace with the actual number of classes


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

# Stack video frames for input to the model
video_data_processed = np.stack(video_frames_data)

num_videos = len(video_frames_data)
# Model Architecture
video_model = models.Sequential([
    layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu'), input_shape=(None, 224, 224, 3)),
    layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
    layers.TimeDistributed(layers.Flatten()),
    layers.LSTM(64, return_sequences=True),
])

output_layer = layers.Dense(num_classes, activation='softmax')(video_model.output[:, -1, :])
# Create the final model
model = models.Model(inputs=video_model.input, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
query_data = ["What is happening in this video?", "Is there a person in the video?"]

# Tokenize and encode query data using BERT tokenizer
tokenized_query = tokenizer(query_data, padding=True, truncation=True, return_tensors="tf")
bert_output_query = bert_model(**tokenized_query)['last_hidden_state'][:, 0, :]

# Make predictions on video frames and queries
predictions = model.predict(video_data_processed)

# Assuming predictions contain the probabilities for each class
# Combine video and query information for final prediction
combined_input = np.concatenate([bert_output_query, predictions], axis=-1)
repeated_queries = np.repeat(bert_output_query, num_videos, axis=0)

# Combine video and query information for final prediction

# Use the combined input to make a final prediction
final_prediction = model.predict(combined_input)

# Convert the prediction to a human-readable answer (you might need a mapping from classes to answers)
# For example, if your classes are [0, 1] and represent [No, Yes]
answer = "Yes" if np.argmax(final_prediction) == 1 else "No"

print("Answer:", answer)
# Train the model with your data
# Replace this with actual training code using your dataset
# model.fit(video_data_processed, labels, epochs=5, batch_size=32)
