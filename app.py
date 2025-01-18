import streamlit as st
import os
import cv2
from pathlib import Path
from ultralytics import YOLO
from PIL import Image

# Initialize the YOLO model
model = YOLO("G:/shieldVision/runs/detect/train2/weights/best.pt")

# Set up the Streamlit UI
st.title("Gun Detection - Upload Image")
st.markdown("Upload an image to detect guns.")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Save the uploaded file
    image_path = os.path.join("uploads", uploaded_file.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    
    # Perform inference on the uploaded image
    results_inference = model.predict(image_path, save=True, imgsz=640)
    
    # Get the saved image with predictions
    result = results_inference[0]
    saved_image_path = Path(result.save_dir) / Path(uploaded_file.name)
    
    # Display the result image
    result_image = Image.open(saved_image_path)
    st.image(result_image, caption="Detection Result", use_container_width=True)
    
    st.success("Detection complete!")
