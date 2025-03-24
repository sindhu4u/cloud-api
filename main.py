from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI()

# Load YOLO model (.pt format)
model = YOLO("best.pt")  # Make sure this file is in the same directory or specify path correctly

@app.get("/")
def read_root():
    return {"message": "YOLO Model API is up!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(img)

    predictions = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            class_id = int(box.cls[0])

            if conf > 0.5:
                predictions.append({
                    "class_id": class_id,
                    "confidence": round(conf, 2),
                    "box": [x1, y1, x2, y2]
                })

    return JSONResponse(content={"predictions": predictions})
