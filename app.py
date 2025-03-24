from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import cv2
import torch
import shutil
import os
from ultralytics import YOLO
import uuid

app = FastAPI()

# Load YOLO model (ensure 'best.pt' is in the same directory)
model_path = os.path.join(os.getcwd(), "best.pt")
model = YOLO(model_path)

@app.post("/detect/")
async def detect_people(video: UploadFile = File(...)):
    temp_filename = f"temp_{uuid.uuid4()}.mp4"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    cap = cv2.VideoCapture(temp_filename)
    output_filename = f"output_{uuid.uuid4()}.mp4"

    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    for results in model(source=temp_filename, stream=True):
        frame = results.orig_img
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = float(box.conf[0])
                if conf > 0.5:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"Person {round(conf, 2)}", (int(x1), int(y1) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()
    os.remove(temp_filename)

    return FileResponse(output_filename, media_type="video/mp4", filename="processed_video.mp4")
