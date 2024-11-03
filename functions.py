from fastapi import FastAPI, UploadFile, HTTPException
import numpy as np
import cv2
import dlib
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import asyncio
# Initialize dlib's face predictor and recognition model
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_reco_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Function to compute embeddings for a single image
def compute_embeddings(img):
    height, width, _ = img.shape
    rect = dlib.rectangle(0, 0, width, height)  # Use full image dimensions
    shape = predictor(img, rect)
    face_descriptor = face_reco_model.compute_face_descriptor(img, shape)
    return face_descriptor

async def get_dlib_embeddings(image_files):
    embeddings = []

    # Function to process a single image file
    async def process_image_file(image_file):
        try:
            image_data = await image_file.read()
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                return None

            # Prepare multiple resized versions (fewer than before to optimize)
            sizes = [92, 150, 400]  # Only resizing to 3 sizes instead of 6
            resized_images_cv = [
                cv2.cvtColor(
                    np.array(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).resize((size, size), Image.LANCZOS)), 
                    cv2.COLOR_RGB2BGR
                ) 
                for size in sizes
            ]

            # Process original and resized images
            images_to_process = [image] + resized_images_cv
            local_embeddings = [compute_embeddings(img) for img in images_to_process]
            
            await image_file.seek(0)
            return local_embeddings
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error processing image {image_file.filename}: {e}"
            )

    # Use ThreadPoolExecutor for parallel processing of images if there are many images
    with ThreadPoolExecutor() as executor:
        tasks = [process_image_file(image_file) for image_file in image_files]
        results = await asyncio.gather(*tasks)

    # Collect embeddings from the results
    for result in results:
        if result:
            embeddings.extend(result)

    if embeddings:
        # Compute the average embedding if multiple embeddings were created
        avg_embedding = np.array(embeddings, dtype=object).mean(axis=0)
        return ','.join(map(str, avg_embedding))
    else:
        raise HTTPException(
            status_code=400, detail="No embeddings could be generated from the provided images."
        )
