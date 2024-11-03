import pandas as pd
import numpy as np
import cv2
import dlib
from ultralytics import YOLO
from pymongo import MongoClient
import asyncio
import uuid
import base64
import os
import torch
import csv
import time
from pymongo.errors import PyMongoError
import json
import requests  
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# MongoDB configuration
MONGO_URI = "mongodb://82.112.231.98:27017/"
client = MongoClient(
    MONGO_URI,
    serverSelectionTimeoutMS=50000,
    socketTimeoutMS=50000,
    connectTimeoutMS=50000,
    maxPoolSize=100
)
db = client["face_database"]
registered_data = db["registered_faces"]
unknown_data = db["unknown_faces"]
# Global variable to track the last save time for unknown visitors
last_unknown_save_time = 0

# Model and device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
yolo_detector = YOLO('yolov8n-face.pt')
# Initialize Dlib's face detector
detector = dlib.get_frontal_face_detector()
# Initialize dlib's face predictor and recognition model
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_reco_model = dlib.face_recognition_model_v1(
    "dlib_face_recognition_resnet_model_v1.dat")

# Path to CSV file
csv_file = "visitor_embeddings.csv"

# Global dictionary to store unknown face data with timestamps and embedding history
unknown_face_cache = {
    'last_saves': {},  # Stores last save time for each face
    'embedding_history': {},  # Stores historical embeddings for comparison
}

# --------------------------- Helper Functions ---------------------------
def calculate_embedding_similarity(embedding1, embedding2):
    """Calculate cosine similarity between two embeddings."""
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

def is_similar_face(new_embedding, stored_embeddings, similarity_threshold=0.95):
    """Check if a new embedding is similar to any stored embeddings."""
    for stored_embedding in stored_embeddings:
        if calculate_embedding_similarity(new_embedding, stored_embedding) > similarity_threshold:
            return True
    return False

def parse_embeddings(embeddings_str: str) -> tuple:
    """Parse string of embeddings into ID and embedding list format."""
    try:
        # First split to separate ID from embeddings
        parts = embeddings_str.split(',')
        if len(parts) < 129:  # ID + 128 embedding values
            print(f"Invalid embedding format: insufficient values")
            return None, []
            
        visitor_id = parts[0]
        # Take exactly 128 values for embeddings
        embeddings = [float(x) for x in parts[1:129]]
        
        return visitor_id, embeddings
    except Exception as e:
        print(f"Error parsing embeddings: {embeddings_str}. Error: {e}")
        return None, []

def fetch_and_write_embeddings_to_csv(csv_file_path: str):
    """Fetch embeddings from MongoDB and append only new embeddings to CSV."""
    try:
        # Check if file exists and get existing IDs
        existing_embeddings = set()
        if os.path.exists(csv_file_path):
            with open(csv_file_path, mode='r', newline='') as file:
                reader = csv.reader(file)
                existing_embeddings = {row[0] for row in reader if row}

        # Get new embeddings from MongoDB
        new_embeddings = []
        documents = registered_data.find({}, {"visitor_id": 1, "embeddings": 1, "_id": 0})
        
        for doc in documents:
            if 'embeddings' not in doc or doc['visitor_id'] in existing_embeddings:
                continue
                
            visitor_id, embeddings = parse_embeddings(doc['embeddings'])
            if visitor_id and len(embeddings) == 128:
                new_embeddings.append([visitor_id] + embeddings)

        # Write new embeddings to CSV
        if new_embeddings:
            # If file doesn't exist, create it with header
            if not os.path.exists(csv_file_path):
                with open(csv_file_path, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    header = ['id'] + [f'embedding_{i}' for i in range(128)]
                    writer.writerow(header)
            
            # Append new embeddings
            with open(csv_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(new_embeddings)
            print(f"Appended {len(new_embeddings)} new embeddings to CSV.")
        else:
            print("No new embeddings to append.")
            
    except Exception as e:
        print(f"Error writing to CSV: {e}")

def load_embeddings_from_csv():
    """Load embeddings from CSV and return face IDs and embeddings."""
    try:
        if not os.path.exists(csv_file):
            print("CSV file not found.")
            return [], np.array([])
            
        # Read CSV with explicit column count
        known_face_data = pd.read_csv(csv_file, header=0)  # Assume first row is header
        
        if len(known_face_data.columns) != 129:  # ID + 128 embedding values
            print(f"Invalid CSV format: expected 129 columns, got {len(known_face_data.columns)}")
            return [], np.array([])
            
        print(f"Loaded {len(known_face_data)} known faces from CSV.")
        return known_face_data.iloc[:, 0].values.tolist(), known_face_data.iloc[:, 1:].values.astype(np.float32)
        
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return [], np.array([])

async def save_unknown_visitor(face_img, embeddings):
    """
    Store unknown visitor's image and embeddings in the unknown_faces collection
    with a strict one-minute save policy for unknown faces.
    """
    global last_unknown_save_time  # Use the global variable for tracking save time
    similarity_threshold = 0.60  # 60% similarity threshold

    try:
        current_time = time.time()
        embeddings_array = np.array(embeddings)

        # Skip saving if the last save was under one minute ago
        if (current_time - last_unknown_save_time) < 60:
            # Check if the new embedding is similar to any in the embedding history
            embedding_history = unknown_face_cache.get('embedding_history', {})
            for _, stored_embeddings in embedding_history.items():
                if is_similar_face(embeddings_array, stored_embeddings, similarity_threshold=similarity_threshold):
                    print("Similar face detected within threshold. Skipping save.")
                    return None, None, None

        _, buffer = cv2.imencode('.jpg', face_img)
        face_bytes = buffer.tobytes()
        face_base64 = base64.b64encode(face_bytes).decode('utf-8')
        new_visitor_id = str(uuid.uuid4())

        # Format embeddings with visitor ID
        embeddings_with_id = f"{new_visitor_id},{','.join(map(str, embeddings))}"

        unknown_record = {
            "visitor_id": new_visitor_id,
            "face_image": face_base64,
            "embeddings": embeddings_with_id,
            "timestamp": current_time
        }

        # Store in the database
        await asyncio.to_thread(unknown_data.insert_one, unknown_record)
        print(f"Saved unknown visitor with ID: {new_visitor_id}")

        # Update the last save time for unknown visitors
        last_unknown_save_time = current_time

        # Add new embedding to the cache for future comparisons
        unknown_face_cache['embedding_history'][new_visitor_id] = [embeddings_array]

        return new_visitor_id, face_base64, embeddings_with_id

    except PyMongoError as e:
        print(f"Error saving unknown visitor: {e}")
        return None, None, None


async def get_dlib_embeddings(image_frames):
    embeddings = []

    for image in image_frames:
        # Ensure the image is in the correct format (BGR to RGB)
        if image is None:
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces in each image using Dlib detector
        faces = detector(image_rgb, 2)
        if len(faces) == 0:
            print("No faces detected.")
            continue
        
        # If faces are detected, compute embeddings
        for face in faces:
            try:
                shape = predictor(image_rgb, face)
                face_descriptor = face_reco_model.compute_face_descriptor(image_rgb, shape)
                embeddings.append(face_descriptor)
            except Exception as e:
                print(f"Error computing face descriptor: {e}")

    if embeddings:
        avg_embedding = np.mean(np.array(embeddings, dtype=np.float32), axis=0)  # Use np.float32 for consistency
        return avg_embedding.tolist()  # Return as a list
    else:
        print("No embeddings were computed.")
        return None  # Explicitly return None if no embeddings are found

def recognize_face(face_descriptor, face_ids, face_embeddings, tolerance=0.40):
    """Recognize a face by comparing its descriptor with known face embeddings."""
    if face_embeddings.size == 0 or face_descriptor is None:
        return "Unknown", 0.0  # Handle the case where embeddings are empty or invalid

    distances = np.linalg.norm(face_embeddings - face_descriptor, axis=1)
    best_match_index = np.argmin(distances)
    best_distance = distances[best_match_index]

    if best_distance <= tolerance:
        return face_ids[best_match_index], 1.0 - (best_distance / tolerance)  # Confidence score based on distance
    else:
        return "Unknown", 0.0  # No match found
    
    
    
    
    # -----------------------Payload to webhook ------------------------
    
    
def create_payload(recognized_visitors, unknown_visitors, community_id, camera_id):
    """Create a JSON payload for recognized and unknown visitors with community and camera IDs."""
    payload = {
        "community_id": community_id,
        "camera_id": camera_id,
        "recognized_visitors": [],
        "unknown_visitors": []
    }

    for visitor in recognized_visitors:
        # Convert face image to base64
        face_img = visitor["face_image"]
        _, buffer = cv2.imencode('.jpg', face_img)
        face_bytes = buffer.tobytes()
        face_base64 = base64.b64encode(face_bytes).decode('utf-8')
        
        payload["recognized_visitors"].append({
            "id": visitor["name"],
            "face_image": face_base64,
            "community_id": community_id,
            "camera_id": camera_id
        })

    for unknown in unknown_visitors:
        payload["unknown_visitors"].append({
            "visitor_id": unknown["visitor_id"],
            "face_image": unknown["face_image"],
            "community_id": community_id,
            "camera_id": camera_id
        })

    return json.dumps(payload)



# Add these global variables at the top of your script
webhook_cache = {
    'last_webhooks': {},  # Store last webhook time for each face (recognized or unknown)
    'last_embeddings': {}  # Store last embeddings for each face
}

def should_send_webhook(face_id, new_embedding, similarity_threshold=0.60, time_threshold_minutes=1):
    """
    Determine if a webhook should be sent based on embedding similarity and time threshold.
    
    Args:
        face_id: Identifier for the face (visitor_id for unknown, name for recognized)
        new_embedding: New face embedding
        similarity_threshold: Minimum similarity threshold (default: 0.60 or 60%)
        time_threshold_minutes: Minimum time between webhooks for similar faces (default: 1 minute)
    
    Returns:
        bool: Whether to send the webhook
    """
    current_time = datetime.now()
    
    # Get last webhook time and embedding for this face
    last_webhook_time = webhook_cache['last_webhooks'].get(face_id)
    last_embedding = webhook_cache['last_embeddings'].get(face_id)
    
    # If no previous webhook for this face, always send
    if last_webhook_time is None or last_embedding is None:
        webhook_cache['last_webhooks'][face_id] = current_time
        webhook_cache['last_embeddings'][face_id] = new_embedding
        return True
    
    # Check time threshold
    time_diff = current_time - last_webhook_time
    if time_diff < timedelta(minutes=time_threshold_minutes):
        # Within time threshold, check similarity
        similarity = calculate_embedding_similarity(new_embedding, last_embedding)
        
        if similarity >= similarity_threshold:
            # Too similar and too recent, don't send webhook
            return False
    
    # Update cache with new data
    webhook_cache['last_webhooks'][face_id] = current_time
    webhook_cache['last_embeddings'][face_id] = new_embedding
    return True

async def recognize_person(frame, face_ids, face_embeddings, community_id, camera_id):
    """Modified recognize_person function with similarity-based webhook sending."""
    results = yolo_detector(frame)
    recognized_visitors = []
    recognized_ids = []
    unknown_visitors_data = []

    for result in results:
        for bbox in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, bbox)
            face_img = frame[y1:y2, x1:x2]
            face_embedding = await get_dlib_embeddings([face_img])

            if face_embedding is None:
                print("No valid face embedding found, skipping this frame.")
                continue

            matched_id, confidence_score = recognize_face(
                face_descriptor=face_embedding,
                face_ids=face_ids,
                face_embeddings=face_embeddings
            )

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Recognized: {matched_id}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Check if we should send webhook based on similarity
            should_send = should_send_webhook(
                face_id=matched_id,
                new_embedding=face_embedding
            )

            if not should_send:
                print(f"Skipping webhook for {matched_id} due to similarity threshold")
                continue

            if matched_id == "Unknown":
                # Save unknown visitor
                new_visitor_id, face_base64, embeddings_with_id = await save_unknown_visitor(
                    face_img,
                    face_embedding
                )
                if new_visitor_id:
                    unknown_visitors_data.append({
                        "visitor_id": new_visitor_id,
                        "face_image": face_base64,
                        "embeddings": embeddings_with_id
                    })
            else:
                recognized_ids.append(matched_id)
                embeddings_with_id = f"{matched_id},{','.join(map(str, face_embedding))}"

            recognized_visitors.append({
                "name": matched_id,
                "face_image": face_img,
                "embeddings": embeddings_with_id,
                "bbox": (x1, y1, x2, y2)
            })

    # Only create and send payload if we have new visitors to report
    if recognized_visitors or unknown_visitors_data:
        payload = create_payload(recognized_visitors, unknown_visitors_data, community_id, camera_id)
        print("Sending payload for new or significantly different faces...")

        try:
            response = requests.post(
                "https://mysentri.com/api/v2/facial-recognition-alerts-webhook",
                data=payload,
                headers={"Content-Type": "application/json"}
            )
            if response.status_code == 200:
                print("Payload sent successfully!")
            else:
                print(f"Failed to send payload: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Error sending webhook: {e}")

    return frame, recognized_visitors, recognized_ids, unknown_visitors_data


# ----------------main function------------------------

async def FaceRecognitionRtsp(rtsp_url, community_id, camera_id):
    """Main function to perform real-time face recognition on an RTSP stream."""
    fetch_and_write_embeddings_to_csv(csv_file)
    face_ids, face_embeddings = load_embeddings_from_csv()

    cap = cv2.VideoCapture(rtsp_url)
    all_recognized_ids = []
    all_unknown_visitors_data = []

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to grab frame from {rtsp_url}")
                break

            frame, recognized_visitors, recognized_ids, unknown_visitors_data = await recognize_person(
                frame, face_ids, face_embeddings, community_id, camera_id
            )

            if recognized_ids:
                all_recognized_ids.extend(recognized_ids)
            all_unknown_visitors_data.extend(unknown_visitors_data)

    except Exception as e:
        print(f"Error in face recognition loop for {rtsp_url}: {e}")
    finally:
        cap.release()
        # cv2.destroyAllWindows()

    return frame, all_recognized_ids,recognized_visitors, all_unknown_visitors_data

async def main(rtsp_urls, community_id, camera_id):
    """Run face recognition concurrently on multiple RTSP URLs."""
    with ThreadPoolExecutor(max_workers=len(rtsp_urls)) as executor:
        loop = asyncio.get_running_loop()
        tasks = [
            loop.run_in_executor(
                executor,
                lambda url=rtsp_url: asyncio.run(FaceRecognitionRtsp(url, community_id, camera_id))
            ) for rtsp_url in rtsp_urls
        ]
        results = await asyncio.gather(*tasks)
        for i, (recognized_ids, unknown_visitors) in enumerate(results):
            print(f"Stream {rtsp_urls[i]}: Final recognized IDs: {recognized_ids}")
            print(f"Stream {rtsp_urls[i]}: Total unknown visitors detected: {len(unknown_visitors)}")

if __name__ == "__main__":
    try:
        rtsp_urls = [
            "rtsp://192.168.1.2:554/stream1",
            "rtsp://192.168.1.3:554/stream2",
            # Add more RTSP URLs as needed
        ]
        community_id = "your_community_id"
        camera_id = "your_camera_id"
        print("Starting face recognition system on multiple streams...")
        asyncio.run(main(rtsp_urls, community_id, camera_id))
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    except Exception as e:
        print(f"Program error: {e}")
    finally:
        # cv2.destroyAllWindows()
        client.close()

