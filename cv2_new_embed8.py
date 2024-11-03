from fastapi import FastAPI, UploadFile, HTTPException
import numpy as np
import cv2
import dlib

# Initialize Dlib's face detector
detector = dlib.get_frontal_face_detector()
# Initialize dlib's face predictor and recognition model
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_reco_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

async def get_dlib_embeddings(image_files):
    embeddings = []

    for image_file in image_files:
        try:
            image_data = await image_file.read()
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                continue

            # Detect faces in the image using Dlib detector
            faces = detector(image)

            if len(faces) != 0:
                for face_rect in faces:
                    # Compute the embedding for each detected face
                    shape = predictor(image, face_rect)
                    face_descriptor = face_reco_model.compute_face_descriptor(image, shape)
                    embeddings.append(face_descriptor)
            else:
                print("No faces detected")

            await image_file.seek(0)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing image {image_file.filename}: {e}")

    if embeddings:
        avg_embedding = np.array(embeddings, dtype=object).mean(axis=0)
        print("Embedding: -> ", avg_embedding)
        return ','.join(map(str, avg_embedding))
    else:
        raise HTTPException(status_code=400, detail="No embeddings could be generated from the provided images.")

# The rest of your FastAPI app and routes would go here...
