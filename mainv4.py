from fastapi import FastAPI, HTTPException, Form, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from pymongo import MongoClient
from pydantic import BaseModel
import dlib
import base64
import asyncio
import io
import time
import cv2
import numpy as np
from bson import ObjectId
from typing import Optional, List, Dict
import logging
from contextlib import asynccontextmanager

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Store active monitoring tasks
active_monitors: Dict[str, asyncio.Task] = {}

# MongoDB Configuration
MONGO_URI = "mongodb://82.112.231.98:27017/"
client = MongoClient(MONGO_URI)
db = client["face_database"]
visitor_data = db["registered_faces"]
unknown_visitor_data = db["unknown_faces"]

# Initialize face recognition components
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_rec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Cache configuration
recent_visitors = {}
CACHE_EXPIRY_SECONDS = 600

class MonitoringStatus(BaseModel):
    camera_id: str
    status: str
    running: bool

class VisitorModel(BaseModel):
    community_id: str
    resident_id: str
    visitor_id: str
    visitor_name: str
    plate_no: Optional[str]
    embeddings: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Create task storage
    yield
    # Shutdown: Cancel all running monitoring tasks
    for task in active_monitors.values():
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

app = FastAPI(lifespan=lifespan)

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

async def get_dlib_embeddings(image_files: List[UploadFile]) -> str:
    try:
        all_embeddings = []
        
        for image_file in image_files:
            # Read image file
            contents = await image_file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                continue
                
            # Convert to RGB (dlib expects RGB images)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            faces = detector(rgb_img)
            
            if len(faces) != 1:
                continue
                
            # Get face landmarks and compute embedding
            shape = predictor(rgb_img, faces[0])
            face_embedding = face_rec.compute_face_descriptor(rgb_img, shape)
            
            # Convert embedding to list and append
            embedding_list = list(face_embedding)
            all_embeddings.append(embedding_list)
        
        if not all_embeddings:
            return ""
        
        # Calculate average embedding
        avg_embedding = np.mean(all_embeddings, axis=0)
        
        # Convert to string format
        embedding_str = ",".join(map(str, avg_embedding))
        return embedding_str
        
    except Exception as e:
        logger.error(f"Error in get_dlib_embeddings: {str(e)}")
        return ""

class FaceRecognitionRtsp:
    def __init__(self, rtsp_url: str, community_id: str, camera_id: str):
        self.rtsp_url = rtsp_url
        self.community_id = community_id
        self.camera_id = camera_id
        self.detector = detector
        self.predictor = predictor
        self.face_rec = face_rec
        
    async def process_frame(self, frame):
        try:
            # Convert frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            faces = self.detector(rgb_frame)
            
            recognized_ids = []
            unknown_visitors = []
            
            for face in faces:
                # Get face landmarks and compute embedding
                shape = self.predictor(rgb_frame, face)
                face_embedding = self.face_rec.compute_face_descriptor(rgb_frame, shape)
                
                # Compare with registered faces in database
                # This is a simplified version - you'll need to implement your matching logic
                matched = False
                for registered_face in visitor_data.find():
                    # Compare embeddings
                    if self.compare_embeddings(face_embedding, registered_face["embeddings"]):
                        recognized_ids.append(registered_face["visitor_id"])
                        matched = True
                        break
                
                if not matched:
                    # Store unknown face
                    face_img = frame[face.top():face.bottom(), face.left():face.right()]
                    unknown_visitors.append({
                        "face_image": self.encode_image(face_img),
                        "timestamp": time.time(),
                        "camera_id": self.camera_id,
                        "community_id": self.community_id
                    })
            
            return frame, recognized_ids, unknown_visitors
            
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return frame, [], []

    @staticmethod
    def compare_embeddings(embedding1, embedding2, threshold=0.6):
        # Convert string embedding to numpy array
        if isinstance(embedding2, str):
            embedding2 = np.array([float(x) for x in embedding2.split(',')])
        
        # Calculate Euclidean distance
        diff = np.array(embedding1) - embedding2
        dist = np.sqrt(np.sum(diff ** 2))
        return dist < threshold

    @staticmethod
    def encode_image(image):
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')

async def rtsp_monitoring_task(rtsp_urls: List[str], community_id: str, camera_id: str):
    """Background task for RTSP monitoring"""
    try:
        logger.info(f"Starting RTSP monitoring for camera {camera_id}")
        
        while True:
            try:
                for rtsp_url in rtsp_urls:
                    cap = cv2.VideoCapture(rtsp_url)
                    
                    if not cap.isOpened():
                        logger.error(f"Failed to open RTSP stream: {rtsp_url}")
                        continue
                    
                    ret, frame = cap.read()
                    if not ret:
                        logger.error(f"Failed to read frame from RTSP stream: {rtsp_url}")
                        continue
                    
                    # Initialize face recognition processor
                    processor = FaceRecognitionRtsp(rtsp_url, community_id, camera_id)
                    
                    # Process frame
                    processed_frame, recognized_ids, unknown_visitors = await processor.process_frame(frame)
                    
                    # Store unknown visitors in database
                    if unknown_visitors:
                        await asyncio.to_thread(unknown_visitor_data.insert_many, unknown_visitors)
                    
                    # Update recent visitors cache
                    for visitor_id in recognized_ids:
                        recent_visitors[visitor_id] = time.time()
                    
                    cap.release()
                
                await asyncio.sleep(0.1)  # Small delay to prevent CPU overload
                
            except Exception as e:
                logger.error(f"Error in RTSP monitoring loop: {str(e)}")
                await asyncio.sleep(5)  # Wait before retrying
                
    except asyncio.CancelledError:
        logger.info(f"RTSP monitoring cancelled for camera {camera_id}")
        raise
    except Exception as e:
        logger.error(f"Fatal error in RTSP monitoring: {str(e)}")
        raise

@app.post("/monitor-multiple-rtsp/")
async def monitor_multiple_rtsp(
    rtsp_urls: List[str] = Form(...),
    community_id: str = Form(...),
    camera_id: str = Form(...),
):
    try:
        # Cancel existing monitoring task for this camera if it exists
        if camera_id in active_monitors:
            if not active_monitors[camera_id].done():
                active_monitors[camera_id].cancel()
                try:
                    await active_monitors[camera_id]
                except asyncio.CancelledError:
                    pass

        # Create and store new monitoring task
        task = asyncio.create_task(
            rtsp_monitoring_task(rtsp_urls, community_id, camera_id)
        )
        active_monitors[camera_id] = task

        return JSONResponse({
            "message": "RTSP monitoring started successfully",
            "camera_id": camera_id,
            "status": "running"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start RTSP monitoring: {str(e)}")

@app.post("/stop-monitoring/{camera_id}")
async def stop_monitoring(camera_id: str):
    """Stop monitoring for a specific camera"""
    if camera_id in active_monitors:
        if not active_monitors[camera_id].done():
            active_monitors[camera_id].cancel()
            try:
                await active_monitors[camera_id]
            except asyncio.CancelledError:
                pass
        del active_monitors[camera_id]
        return JSONResponse({"message": f"Monitoring stopped for camera {camera_id}"})
    raise HTTPException(status_code=404, detail="No active monitoring found for this camera")

@app.get("/monitoring-status/")
async def get_monitoring_status():
    """Get status of all monitoring tasks"""
    status = []
    for camera_id, task in active_monitors.items():
        status.append(MonitoringStatus(
            camera_id=camera_id,
            status="running" if not task.done() else "stopped",
            running=not task.done()
        ))
    return status

@app.get("/unknown-visitors/")
async def get_unknown_visitors():
    try:
        unknown_visitors = await asyncio.to_thread(list, unknown_visitor_data.find())
        serialized_visitors = [serialize_mongo_document(visitor) for visitor in unknown_visitors]
        return JSONResponse(content={"unknown_visitors": serialized_visitors})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve unknown visitors: {str(e)}")

@app.post("/visitors/register-with-single-image/", response_model=VisitorModel)
async def register_visitor_with_single_image(
    community_id: str = Form(...),
    resident_id: str = Form(...),
    visitor_id: str = Form(...),
    visitor_name: str = Form(...),
    plate_no: str = Form(None),
    image: UploadFile = File(...)
):
    try:
        # Clean up input data
        community_id = strip_extra_quotes(community_id)
        resident_id = strip_extra_quotes(resident_id)
        visitor_id = strip_extra_quotes(visitor_id)
        visitor_name = strip_extra_quotes(visitor_name)
        plate_no = strip_extra_quotes(plate_no)

        # Check for existing visitor
        if visitor_data.find_one({"visitor_id": visitor_id}):
            raise HTTPException(status_code=400, detail="Visitor ID already exists")

        # Process image
        image_bytes = await image.read()
        image_copies = [
            UploadFile(filename=f"copy_{i}.jpg", file=io.BytesIO(image_bytes)) 
            for i in range(5)
        ]

        # Generate embeddings
        embeddings = await get_dlib_embeddings(image_copies)
        if not embeddings:
            raise HTTPException(status_code=400, detail="No valid face detected")

        embeddings_with_id = f"{visitor_id},{embeddings}"

        # Create visitor document
        visitor = {
            "community_id": community_id,
            "resident_id": resident_id,
            "visitor_id": visitor_id,
            "visitor_name": visitor_name,
            "plate_no": plate_no,
            "embeddings": embeddings_with_id,
        }

        # Insert into database
        result = await asyncio.to_thread(visitor_data.insert_one, visitor)
        created_visitor = await asyncio.to_thread(
            visitor_data.find_one, {"_id": result.inserted_id}
        )

        # Prepare response
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
    new_visitor_id: str = Form(...),
    community_id: str = Form(...),
    resident_id: str = Form(...),
    visitor_name: str = Form(...),
    plate_no: Optional[str] = Form(None),
):
    try:
        # Find unknown visitor
        unknown_visitor = await asyncio.to_thread(
            unknown_visitor_data.find_one, {"visitor_id": unknown_visitor_id}
        )

        if not unknown_visitor:
            raise HTTPException(status_code=404, detail="Unknown visitor not found")

        # Update embeddings
        old_embeddings = unknown_visitor["embeddings"]
        updated_embeddings = f"{new_visitor_id},{','.join(old_embeddings.split(',')[1:])}"

        # Create registered visitor document
        registered_visitor = {
            "community_id": community_id,
            "resident_id": resident_id,
            "visitor_id": new_visitor_id,
            "visitor_name": visitor_name,
            "plate_no": plate_no,
            "embeddings": updated_embeddings,
            "face_image": unknown_visitor["face_image"],
        }

        # Insert and delete
        result = await asyncio.to_thread(visitor_data.insert_one, registered_visitor)
        if result.inserted_id:
            await asyncio.to_thread(
                unknown_visitor_data.delete_one, {"visitor_id": unknown_visitor_id}
            )

        return JSONResponse(content={
            "message": "Visitor successfully registered",
            "new_visitor_id": new_visitor_id
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update visitor: {str(e)}")

# Main entry point to run the FastAPI application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("mainv4:app", host="0.0.0.0", port=8000)
