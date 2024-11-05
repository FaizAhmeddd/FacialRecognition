from fastapi import FastAPI, HTTPException, Form, UploadFile, File
from fastapi.responses import JSONResponse
from pymongo import MongoClient
from pydantic import BaseModel
import dlib
import base64
import asyncio
import io
import time
from bson import ObjectId
from typing import Optional, List, Dict
from functions import get_dlib_embeddings
from recognition_with_webcam import FaceRecognitionWebcam
from main_with_rtsp_recognition import FaceRecognitionRtsp
import logging
from datetime import datetime
from typing import Dict
import signal
import uvicorn
import multiprocessing

# --------------------------- MongoDB & detector ------------------------------

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

# Setup logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global dictionary to keep track of running processes
process_registry: Dict[str, multiprocessing.Process] = {}
# Initialize face detector



# ---------------------------------- get unknown visitor records ------------------------------




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

@app.get("/unknown-visitors/")
async def get_unknown_visitors():
    try:
        # Retrieve all records from the unknown_visitors collection
        unknown_visitors = await asyncio.to_thread(list, unknown_visitor_data.find())
        
        # Serialize the documents to make them JSON serializable
        serialized_visitors = [serialize_mongo_document(visitor) for visitor in unknown_visitors]
        
        # Return the serialized data as JSON response
        return JSONResponse(content={"unknown_visitors": serialized_visitors})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve unknown visitors: {str(e)}")

def update_recent_visitor(visitor_id: str):
    recent_visitors[visitor_id] = time.time()


# --------------------------- RTSP Monitoring ------------------------------

def monitor_rtsp_stream(rtsp_url: str, community_id: str, camera_id: str):
    """Function that will run in a separate process to monitor a specific RTSP stream."""
    logger.info(f"Started RTSP monitoring for {rtsp_url} (Community: {community_id}, Camera: {camera_id})")
    
    try:
        while True:
            # Call the async FaceRecognitionRtsp function using asyncio.run
            frame, recognized_ids, recognized_visitors, unknown_visitors = asyncio.run(
                FaceRecognitionRtsp(rtsp_url, community_id, camera_id)
            )
            logger.info(f"Processed stream {rtsp_url} - recognized IDs: {recognized_ids}")
            time.sleep(5)  # Simulate processing delay
    except Exception as e:
        logger.error(f"Error in monitoring RTSP stream {rtsp_url}: {str(e)}")

# Helper function to start RTSP monitoring in a new process
def start_rtsp_monitoring_process(rtsp_urls: List[str], community_id: str, camera_id: str):
    """This function will create separate processes for each RTSP URL."""
    for rtsp_url in rtsp_urls:
        process_key = f"{community_id}_{camera_id}_{rtsp_url}"
        if process_key in process_registry:
            logger.info(f"RTSP monitoring for {rtsp_url} already running. Skipping.")
            continue

        # Create a new process for each RTSP URL
        process = multiprocessing.Process(target=monitor_rtsp_stream, args=(rtsp_url, community_id, camera_id))
        process.start()
        process_registry[process_key] = process

        logger.info(f"Started new process for RTSP monitoring: {rtsp_url} (PID: {process.pid})")

@app.post("/monitor-multiple-rtsp/")
async def monitor_multiple_rtsp(
    rtsp_urls: List[str] = Form(...),
    community_id: str = Form(...),
    camera_id: str = Form(...),
):
    """Start separate processes to monitor multiple RTSP streams concurrently."""
    try:
        # Start new background processes for each RTSP URL
        start_rtsp_monitoring_process(rtsp_urls, community_id, camera_id)
        return JSONResponse({
            "message": "RTSP monitoring started successfully",
            "community_id": community_id,
            "camera_id": camera_id,
            "rtsp_urls": rtsp_urls
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start RTSP monitoring: {str(e)}")

# --------------------------- Webcam Monitoring ------------------------------

def monitor_webcam_process(webcam_index: int, community_id: str, camera_id: str):
    """Run webcam monitoring in a separate process."""
    try:
        recognized_ids, unknown_visitors = asyncio.run(FaceRecognitionWebcam(webcam_index, community_id, camera_id))
        for visitor_id in recognized_ids:
            logger.info(f"Visitor {visitor_id} recognized.")
    except Exception as e:
        logger.error(f"Failed to monitor webcam: {str(e)}")

@app.post("/monitor-webcam/")
async def monitor_webcam(
    community_id: str = Form(...),
    camera_id: str = Form(...),
    webcam_index: int = Form(0),
):
    """Start webcam monitoring in a background process."""
    try:
        process = multiprocessing.Process(target=monitor_webcam_process, args=(webcam_index, community_id, camera_id))
        process.start()
        process_registry[f"webcam_{community_id}_{camera_id}"] = process

        return JSONResponse({"message": "Webcam monitoring started successfully"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start webcam monitoring: {str(e)}")

# --------------------------- Register Visitor with Image ------------------------------

def register_visitor_with_image(community_id, resident_id, visitor_id, visitor_name, plate_no, image_bytes):
    """Process the visitor registration with the uploaded image in a separate process, ensuring unique visitor_id."""
    try:
        # Check if the visitor_id already exists in the registered_faces collection
        if visitor_data.find_one({"visitor_id": visitor_id}):
            logger.error(f"Visitor ID {visitor_id} already exists in the registered_faces collection.")
            return {"error": f"Visitor ID {visitor_id} already exists."}

        # Generate face embeddings using the uploaded image
        image_copies = [UploadFile(filename=f"copy_{i}.jpg", file=io.BytesIO(image_bytes)) for i in range(5)]
        embeddings = asyncio.run(get_dlib_embeddings(image_copies))

        if not embeddings:
            logger.error("No valid face embeddings could be generated.")
            return {"error": "No valid face embeddings could be generated."}

        # Prepare the document to insert into MongoDB
        embeddings_with_id = f"{visitor_id},{embeddings}"
        visitor = {
            "community_id": community_id,
            "resident_id": resident_id,
            "visitor_id": visitor_id,
            "visitor_name": visitor_name,
            "plate_no": plate_no,
            "embeddings": embeddings_with_id,
        }

        # Insert the new visitor record into the database
        result = visitor_data.insert_one(visitor)
        logger.info(f"Visitor {visitor_id} registered successfully with ID {result.inserted_id}")
        return {"status": "success", "visitor_id": visitor_id}

    except Exception as e:
        logger.error(f"Failed to register visitor: {str(e)}")
        return {"error": f"Failed to register visitor: {str(e)}"}



@app.post("/visitors/register-with-single-image/")
async def register_visitor_with_single_image(
    community_id: str = Form(...),
    resident_id: str = Form(...),
    visitor_id: str = Form(...),
    visitor_name: str = Form(...),
    plate_no: str = Form(None),
    image: UploadFile = File(...),
):
    """Start visitor registration process in a background process, ensuring unique visitor_id."""
    try:
        # Read the uploaded image once
        image_bytes = await image.read()

        # Start the registration process in a new background process
        process = multiprocessing.Process(
            target=register_visitor_with_image,
            args=(community_id, resident_id, visitor_id, visitor_name, plate_no, image_bytes)
        )
        process.start()
        process_registry[f"register_{visitor_id}"] = process

        return JSONResponse({"message": f"Visitor registration process started for {visitor_id}"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start visitor registration process: {str(e)}")


# --------------------------- Update Unknown Visitor to Registered ------------------------------

def update_unknown_to_registered(unknown_visitor_id, new_visitor_id, community_id, resident_id, visitor_name, plate_no):
    """Update an unknown visitor to registered in a background process and remove the record from the unknown collection. Ensures visitor_id uniqueness."""
    try:
        # Check if the new_visitor_id already exists in the registered_faces collection
        if visitor_data.find_one({"visitor_id": new_visitor_id}):
            logger.error(f"Visitor ID {new_visitor_id} already exists in the registered_faces collection.")
            return {"error": f"Visitor ID {new_visitor_id} already exists."}

        # Find the visitor in the unknown_faces collection
        unknown_visitor = unknown_visitor_data.find_one({"visitor_id": unknown_visitor_id})
        if not unknown_visitor:
            logger.error(f"Unknown visitor with ID {unknown_visitor_id} not found.")
            return {"error": "Unknown visitor not found."}

        # Update the embeddings and prepare the document for registered_faces
        old_embeddings = unknown_visitor["embeddings"]
        updated_embeddings = f"{new_visitor_id},{','.join(old_embeddings.split(',')[1:])}"

        registered_visitor = {
            "community_id": community_id,
            "resident_id": resident_id,
            "visitor_id": new_visitor_id,
            "visitor_name": visitor_name,
            "plate_no": plate_no,
            "embeddings": updated_embeddings,
            "face_image": unknown_visitor["face_image"],  # Keep the face image from the unknown record
        }

        # Insert the updated visitor into the registered_faces collection
        result = visitor_data.insert_one(registered_visitor)

        if result.inserted_id:
            # After a successful insertion, delete the visitor from unknown_faces collection
            delete_result = unknown_visitor_data.delete_one({"visitor_id": unknown_visitor_id})
            if delete_result.deleted_count == 1:
                logger.info(f"Unknown visitor {unknown_visitor_id} successfully updated to registered visitor {new_visitor_id} and removed from unknown_faces collection.")
            else:
                logger.error(f"Failed to remove unknown visitor {unknown_visitor_id} from the unknown_faces collection.")
            
        return {"status": "success", "visitor_id": new_visitor_id}

    except Exception as e:
        logger.error(f"Failed to update unknown visitor {unknown_visitor_id}: {str(e)}")
        return {"error": f"Failed to update visitor: {str(e)}"}



@app.put("/unknown-visitors/register/")
async def update_unknown_visitor_to_registered(
    unknown_visitor_id: str = Form(...),
    new_visitor_id: str = Form(...),
    community_id: str = Form(...),
    resident_id: str = Form(...),
    visitor_name: str = Form(...),
    plate_no: Optional[str] = Form(None),
):
    """Start the process to update unknown visitor to registered in a background process."""
    try:
        # Start the update process in a new background process
        process = multiprocessing.Process(
            target=update_unknown_to_registered,
            args=(unknown_visitor_id, new_visitor_id, community_id, resident_id, visitor_name, plate_no)
        )
        process.start()
        process_registry[f"update_{new_visitor_id}"] = process

        return JSONResponse({"message": f"Update process started for unknown visitor {unknown_visitor_id}"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start update process: {str(e)}")


# --------------------------- Shutdown Logic ------------------------------

@app.on_event("shutdown")
async def shutdown_event():
    """Gracefully terminate all running processes on shutdown."""
    logger.info("Shutting down application...")

    # Terminate all running processes
    for key, process in list(process_registry.items()):
        if process.is_alive():
            logger.info(f"Terminating process {key} (PID: {process.pid})")
            try:
                # Attempt to terminate the process gracefully
                process.terminate()
                process.join(timeout=2)  # Give the process some time to terminate
            except Exception as e:
                logger.error(f"Error terminating process {key} (PID: {process.pid}): {str(e)}")

            # If the process is still alive after termination, force kill it
            if process.is_alive():
                logger.info(f"Force killing process {key} (PID: {process.pid})")
                try:
                    process.kill()  # Force kill if it didn't terminate
                    process.join()  # Ensure the process is fully terminated
                except Exception as e:
                    logger.error(f"Error killing process {key} (PID: {process.pid}): {str(e)}")

        # Remove the process from the registry once handled
        process_registry.pop(key, None)

    logger.info("All processes terminated. Application shutdown complete.")

# --------------------------- Main Entry Point ------------------------------

if __name__ == "__main__":
    # Use `uvicorn` to run the FastAPI app
    uvicorn.run("mainv4:app", host="0.0.0.0", port=8000)
