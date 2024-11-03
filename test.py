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
import re
import torch
import csv
import time
from pymongo.errors import PyMongoError
import json
import requests

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
    global last_unknown_save_time

    try:
        current_time = time.time()
        embeddings_array = np.array(embeddings)

        # Convert embeddings to a stable format for comparison
        embedding_key = hash(tuple(np.round(embeddings_array, decimals=5)))

        # Skip saving if the last save was under one minute ago
        if (current_time - last_unknown_save_time) < 60:
            print(f"Skipping save - Only {current_time - last_unknown_save_time:.1f} seconds since last save.")
            return None, None, None

        # Prepare to save the new visitor
        face_img_contiguous = np.ascontiguousarray(face_img)
        face_base64 = base64.b64encode(face_img_contiguous).decode('utf-8')
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

        return new_visitor_id, face_base64, embeddings_with_id

    except PyMongoError as e:
        print(f"Error saving unknown visitor: {e}")
        return None, None, None

async def get_dlib_embeddings(image_frames):
    """Get face embeddings using dlib."""
    embeddings = []

    for image in image_frames:
        if image is None:
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        faces = detector(image_rgb, 2)
        if len(faces) == 0:
            print("No faces detected.")
            continue
        
        for face in faces:
            try:
                shape = predictor(image_rgb, face)
                face_descriptor = face_reco_model.compute_face_descriptor(image_rgb, shape)
                embeddings.append(face_descriptor)
            except Exception as e:
                print(f"Error computing face descriptor: {e}")

    if embeddings:
        avg_embedding = np.mean(np.array(embeddings, dtype=np.float32), axis=0)
        return avg_embedding.tolist()
    else:
        print("No embeddings were computed.")
        return None

def recognize_face(face_descriptor, face_ids, face_embeddings, tolerance=0.40):
    """Recognize a face by comparing its descriptor with known face embeddings."""
    if face_embeddings.size == 0 or face_descriptor is None:
        return "Unknown", 0.0

    distances = np.linalg.norm(face_embeddings - face_descriptor, axis=1)
    min_distance = np.min(distances)
    confidence_score = max(0, 1 - min_distance / tolerance)
    
    if min_distance < tolerance:
        matched_id = face_ids[np.argmin(distances)]
        return matched_id, confidence_score
    return "Unknown", confidence_score

def create_payload(recognized_visitors, unknown_visitors):
    """Create a JSON payload for recognized and unknown visitors."""
    payload = {
        "recognized_visitors": [],
        "unknown_visitors": []
    }

    for visitor in recognized_visitors:
        payload["recognized_visitors"].append({
            "id": visitor["name"],
            "confidence": visitor["confidence"],
            "bbox": visitor["bbox"],
            "face_image": visitor["face_image"]  # Base64 encoded image for recognized visitors
        })

    for unknown in unknown_visitors:
        payload["unknown_visitors"].append({
            "visitor_id": unknown["visitor_id"],
            "face_image": unknown["face_image"],
            "embeddings": unknown["embeddings"]
        })

    return json.dumps(payload)

async def recognize_person(frame, face_ids, face_embeddings):
    """Recognize persons in frame using YOLO for detection and Dlib for embedding recognition."""
    results = yolo_detector(frame)
    recognized_visitors = []
    recognized_ids = []
    unknown_visitors_data = []

    for result in results:
        for bbox in result.boxes.xyxy:
            face_embedding = await get_dlib_embeddings([frame])

            if face_embedding is None:
                print("No valid face embedding found, skipping this frame.")
                continue

            matched_id, confidence_score = recognize_face(
                face_descriptor=face_embedding,
                face_ids=face_ids,
                face_embeddings=face_embeddings
            )

            x1, y1, x2, y2 = map(int, bbox)
            face_img = frame[y1:y2, x1:x2]  # Extract face image
            face_img_contiguous = np.ascontiguousarray(face_img)
            face_base64 = base64.b64encode(face_img_contiguous).decode('utf-8')

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Recognized: {matched_id} (Confidence: {confidence_score:.2f})"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            if matched_id == "Unknown":
                new_visitor_id, _, embeddings_with_id = await save_unknown_visitor(
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
                recognized_visitors.append({
                    "name": matched_id,
                    "bbox": (x1, y1, x2, y2),
                    "confidence": confidence_score,
                    "face_image": face_base64  # Include face image for recognized visitors
                })

    # Create and send payload
    payload = create_payload(recognized_visitors, unknown_visitors_data)
    print("Payload ready for API call:", payload)

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
        print(f"Error sending payload: {e}")

    return frame, recognized_visitors, recognized_ids, unknown_visitors_data

async def FaceRecognition(webcam_stream):
    """Main function to perform real-time face recognition on a webcam stream."""
    fetch_and_write_embeddings_to_csv(csv_file)
    face_ids, face_embeddings = load_embeddings_from_csv()

    cap = cv2.VideoCapture(webcam_stream)
    all_recognized_ids = []
    all_unknown_visitors_data = []

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            frame, recognized_visitors, recognized_ids, unknown_visitors_data = await recognize_person(
                frame, face_ids, face_embeddings
            )

            cv2.imshow("Webcam Face Recognition", frame)

            if recognized_ids:
                all_recognized_ids.extend(recognized_ids)
            all_unknown_visitors_data.extend(unknown_visitors_data)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("User requested exit")
                break

    except Exception as e:
        print(f"Error in face recognition loop: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
    return all_recognized_ids, all_unknown_visitors_data

if __name__ == "__main__":
    try:
        print("Starting face recognition system...")
        recognized_ids, unknown_visitors = asyncio.run(FaceRecognition(0))
        print(f"Final recognized IDs: {recognized_ids}")
        print(f"Total unknown visitors detected: {len(unknown_visitors)}")
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    except Exception as e:
        print(f"Program error: {e}")
    finally:
        cv2.destroyAllWindows()
        client.close()