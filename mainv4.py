from fastapi import FastAPI, HTTPException, Form, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from pymongo import MongoClient
from pydantic import BaseModel
import dlib
import base64
import asyncio
import io
import time
from bson import ObjectId
from typing import Optional, List
from functions import get_dlib_embeddings
from recognition_with_webcam import FaceRecognitionWebcam
from main_with_rtsp_recognition import FaceRecognitionRtsp
import cv2

# Initialize FastAPI
app = FastAPI()

# MongoDB Configuration
MONGO_URI = "mongodb://82.112.231.98:27017/"
client = MongoClient(MONGO_URI)
db = client["face_database"]
visitor_data = db["registered_faces"]
unknown_visitor_data = db["unknown_faces"]

# Initialize face detector
detector = dlib.get_frontal_face_detector()

# In-memory cache
recent_visitors = {}
CACHE_EXPIRY_SECONDS = 600

def strip_extra_quotes(value: str) -> str:
    if value:
        return value.strip('"')
    return value

def serialize_mongo_document(document):
    if isinstance(document, dict):
        return {k: serialize_mongo_document(v) for k, v in document.items()}
    elif isinstance(document, list):
        return [serialize_mongo_document(item) for item in document]
    elif isinstance(document, ObjectId):
        return str(document)
    else:
        return document

def is_recently_processed(visitor_id: str) -> bool:
    current_time = time.time()
    if visitor_id in recent_visitors:
        last_seen = recent_visitors[visitor_id]
        if current_time - last_seen < CACHE_EXPIRY_SECONDS:
            return True
        else:
            del recent_visitors[visitor_id]
    return False

def update_recent_visitor(visitor_id: str):
    recent_visitors[visitor_id] = time.time()

# Capture webcam stream
async def capture_webcam_stream(webcam_index: int, community_id: str, camera_id: str):
    recognized_ids, unknown_visitors = await FaceRecognitionWebcam(webcam_index, community_id, camera_id)
    
    for visitor_id in recognized_ids:
        # Check if this visitor has been recently processed
        if is_recently_processed(visitor_id):
            print(f"Visitor {visitor_id} already processed recently. Skipping notification.")
            continue  # Skip sending the webhook if already processed

# Start webcam monitoring
async def start_webcam_monitoring(webcam_index: int, community_id: str, camera_id: str):
    await capture_webcam_stream(webcam_index, community_id, camera_id)

@app.post("/monitor-webcam/")
async def monitor_webcam(
    community_id: str = Form(...),
    camera_id: str = Form(...),
    webcam_index: int = Form(0),
):
    try:
        # Directly call the function instead of using background tasks
        await start_webcam_monitoring(webcam_index, community_id, camera_id)
        return JSONResponse({"message": "Webcam monitoring completed successfully"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to monitor webcam: {str(e)}")

# Capture frames from an authenticated RTSP stream
async def capture_rtsp_stream(rtsp_url: str, community_id: str, camera_id: str):
    # Process the frame for face recognition
    frame, all_recognized_ids, recognized_visitors, all_unknown_visitors_data = await FaceRecognitionRtsp(rtsp_url, community_id, camera_id)
    await asyncio.sleep(0.01)  # Add a small delay to avoid overwhelming the CPU

# Start RTSP monitoring (runs indefinitely in the background)
async def start_rtsp_monitoring(rtsp_urls: List[str], community_id: str, camera_id: str):
    while True:
        tasks = [capture_rtsp_stream(url, community_id, camera_id) for url in rtsp_urls]
        await asyncio.gather(*tasks)  # Run all capture tasks concurrently
        await asyncio.sleep(1)  # Add delay between monitoring cycles to avoid overwhelming resources

@app.post("/monitor-multiple-rtsp/")
async def monitor_multiple_rtsp(
    rtsp_urls: List[str] = Form(...),
    community_id: str = Form(...),
    camera_id: str = Form(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    try:
        # Start RTSP monitoring tasks in the background for each URL
        background_tasks.add_task(start_rtsp_monitoring, rtsp_urls, community_id, camera_id)
        return JSONResponse({"message": "RTSP monitoring started successfully for multiple streams"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start RTSP monitoring: {str(e)}")

# API endpoint to register a visitor with a single image
class VisitorModel(BaseModel):
    community_id: str
    resident_id: str
    visitor_id: str
    visitor_name: str
    plate_no: Optional[str]
    embeddings: str

@app.post("/visitors/register-with-single-image/", response_model=VisitorModel)
async def register_visitor_with_single_image(
    community_id: str = Form(...),
    resident_id: str = Form(...),
    visitor_id: str = Form(...),
    visitor_name: str = Form(...),
    plate_no: str = Form(None),  # Optional field
    image: UploadFile = File(...)  # Image file
):
    try:
        # Clean up input data by stripping any extra quotes
        community_id = strip_extra_quotes(community_id)
        resident_id = strip_extra_quotes(resident_id)
        visitor_id = strip_extra_quotes(visitor_id)
        visitor_name = strip_extra_quotes(visitor_name)
        plate_no = strip_extra_quotes(plate_no)

        # Check if the visitor with the same unique ID already exists in the database
        if visitor_data.find_one({"visitor_id": visitor_id}):
            raise HTTPException(status_code=400, detail="Visitor ID already exists and must be unique.")

        if not image:
            raise HTTPException(status_code=400, detail="No image was provided.")

        # Read the uploaded image once
        image_bytes = await image.read()

        # Create 5 copies of the same image
        image_copies = [
            UploadFile(filename=f"copy_{i}.jpg", file=io.BytesIO(image_bytes)) for i in range(5)
        ]

        # Generate face embeddings using the 5 copies of the image
        embeddings = await get_dlib_embeddings(image_copies)

        if not embeddings:
            raise HTTPException(status_code=400, detail="No valid face embeddings could be generated.")

        # Add the visitor ID to the embeddings for identification purposes
        embeddings_with_id = f"{visitor_id},{embeddings}"

        # Create a visitor document to insert into the MongoDB collection
        visitor = {
            "community_id": community_id,
            "resident_id": resident_id,
            "visitor_id": visitor_id,
            "visitor_name": visitor_name,
            "plate_no": plate_no,
            "embeddings": embeddings_with_id,
        }

        # Insert the new visitor record into the database (async)
        result = await asyncio.to_thread(visitor_data.insert_one, visitor)
        created_visitor = await asyncio.to_thread(visitor_data.find_one, {"_id": result.inserted_id})

        # Encode the original image in base64 format to return in the response
        response_data = created_visitor
        response_data['captured_images'] = [
            base64.b64encode(image_bytes).decode('utf-8')
        ]
        
        return serialize_mongo_document(response_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to register visitor: {str(e)}")

@app.put("/unknown-visitors/register/")
async def update_unknown_visitor_to_registered(
    unknown_visitor_id: str = Form(...),
    new_visitor_id: str = Form(...),  # Manually assigned new visitor ID
    community_id: str = Form(...),
    resident_id: str = Form(...),
    visitor_name: str = Form(...),
    plate_no: Optional[str] = Form(None),
):
    try:
        # Find unknown visitor in "unknown_faces" collection by unknown_visitor_id
        unknown_visitor = await asyncio.to_thread(
            unknown_visitor_data.find_one, {"visitor_id": unknown_visitor_id}
        )

        # Raise an error if the visitor record is not found
        if not unknown_visitor:
            raise HTTPException(status_code=404, detail="Unknown visitor not found.")

        # Update embeddings with the new visitor_id
        old_embeddings = unknown_visitor["embeddings"]
        updated_embeddings = f"{new_visitor_id},{','.join(old_embeddings.split(',')[1:])}"

        # Assign fields according to "registered_faces" schema
        registered_visitor = {
            "community_id": community_id,
            "resident_id": resident_id,
            "visitor_id": new_visitor_id,  # New visitor ID
            "visitor_name": visitor_name,
            "plate_no": plate_no,
            "embeddings": updated_embeddings,  # Updated embeddings with new visitor_id
            "face_image": unknown_visitor["face_image"],  # Use face_image from the unknown visitor
        }

        # Insert visitor into the "registered_faces" collection
        result = await asyncio.to_thread(visitor_data.insert_one, registered_visitor)

        # If successfully inserted, delete the unknown visitor record
        if result.inserted_id:
            await asyncio.to_thread(unknown_visitor_data.delete_one, {"visitor_id": unknown_visitor_id})

        return JSONResponse(content={"message": "Visitor successfully updated to registered.", "new_visitor_id": new_visitor_id})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update visitor: {str(e)}")

# Main entry point to run the FastAPI application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("mainv4:app", host="0.0.0.0", port=8000)
