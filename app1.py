import cv2
from ultralytics import YOLO
from PIL import Image
import torch
import time
import numpy as np
import streamlit as st

# Load YOLO model (cached so it doesn't reload on every fragment tick)
@st.cache_resource
def load_model():
    return YOLO("yolov5n.pt")

model = load_model()

# Defining thermodynamics variables:
Room_volume = 5 * 3 * 7  # l*h*b
Average_T = 25  # degC
T_source = 130  # W per human
Load = 200 + 850 + 540 + 700 + 1000  # lights plus computers plus walls plus sun plus ventilation
k = 0.000867  # constant of cooling units: s-1
T_amb = 20  # degC
H_t = 37  # degC

st.title("Smart HVAC system dashboard")
st.subheader("Real time data")
st.write("The web cam is on.")

# Initialize webcam once using session_state so it stays active
if "cap" not in st.session_state or not st.session_state.cap.isOpened():
    cap = cv2.VideoCapture(0, cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 160)
    st.session_state.cap = cap

# Create persistent placeholders outside the fragment
metric_placeholder = st.empty()
image_placeholder = st.empty()

@st.fragment(run_every=0.15)  # Balanced interval to prevent CPU queue buildup
def render_live_metrics():
    cap = st.session_state.cap
    
    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
        return  

    cap.grab()
    ret, frame = cap.read()
    
    if not ret or frame is None:
        st.warning("Waiting for webcam stream...")
        return

    # Explicitly resize frame to match low resolution for speed
    frame = cv2.resize(frame, (160, 160))
    frame = cv2.flip(frame, 1)  
    annotated_frame = frame.copy()
    
    # Run YOLO detection with explicit small image size (imgsz)
    result = model(annotated_frame, imgsz=160, verbose=False)
    frame_count = 0
    for r in result:
        annotated_frame = r.plot()
        frame_count = (r.boxes.cls == 0).sum().item()  # count people (class 0)

    # Put text overlay on frame
    cv2.putText(
        annotated_frame,
        f"People: {frame_count}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )
    
    # Convert color scheme from BGR to RGB for Streamlit
    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    
    # Calculate thermodynamics metrics
    Qdesign = Load + frame_count * T_source
    tr = Qdesign / 3.157  
    P = Qdesign / 3.3     
    
    # Update metrics inside the persistent placeholder container
    with metric_placeholder.container():
        col1, col2, col3 = st.columns(3)
        with col1:  
            st.metric(label="Required Cooling", value=f"{round(tr)} W")
        with col2:
            st.metric(label="Power", value=f"{round(P)} W")
        with col3:
            st.metric(label="People", value=frame_count)
            
    # Update the video frame display dynamically
    image_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

# Run the live loop fragment
render_live_metrics()
