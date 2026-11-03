import cv2
from ultralytics import YOLO
from PIL import Image
import torch
import time
import pyautogui
import numpy as np
import streamlit as st


model = YOLO("yolov5s.pt")
#seen_ids = set()       # every unique person ID seen since the script started
#last_boxes = []        # most recent known (x1, y1, x2, y2, id) boxes, reused between detections
frame_count = 0
# for downloading the video

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
filename = "Recording.mp4"
#defining thermodyanmics variables:
Room_volume= 5*3*7#l*h*b
Average_T = 25# degC
T_source = 130# W per human
Load=200+850+540+700+1000#ligths plus computers plus walls plus sun plus ventilation
k= 0.000867#constant of cooling units: s-1
T_amb = 20# degC
H_t= 37#degC
Qdesign=0
tr=0
P=0

cap = cv2.VideoCapture(0)#0 is opening up webcam
#setting the boundaries, very imp
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_heigth= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
resolution=(frame_width, frame_heigth)
out = cv2.VideoWriter(filename,fourcc,10,resolution)


st.title("Smart HVAC syst  dashboard")
st.subheader("Real time data")
st.write("The web cam is on.")
#st.write(f"Required Cooling Capacity: {tr} W")
#st.write(f"Power Consumption: {P} W")
#st.write(f"People in frame: {frame_count}")
while cap.isOpened():
    placeholder = st.empty()  # Create a placeholder for the live metrics
    #@st.fragment(run_every=1)


    # Render the live section
    #render_live_metrics()
    ret, frame = cap.read()# basically reading the frame
    if not ret:
       print("f  f")
    frame = cv2.flip(frame, 1) # flipping the frame
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = 10# medium accuracy to higher
    #output window pixels or whtever it is.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)





    annotated_frame = frame.copy()
    result = model(annotated_frame, verbose=False)#this verbose prevents it from outputing too much data to terminal
    for r in result:
        annotated_frame = r.plot()# plots a rectangle
        frame_count = (r.boxes.cls == 0).sum().item()# count the no. of people in the frame
    cv2.putText(
                annotated_frame,
                f"People in frame: {frame_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )
    #cv2.imshow('frame',annotated_frame)#outputs the window showing the frames
    #frame = np.array(annotated_frame)# recording the frame in an array 
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)# reversing color to suit the opencv scheme
    #out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break# breaks the ckt
    Qdesign=Load+frame_count*(T_source)
    tr=Qdesign/3.157# required cooling capacity
    P=Qdesign/3.3#3.3 is COP of AC
    with placeholder.container():
        st.write(f"Required Cooling Capacity: {tr} W")
        st.write(f"Power Consumption: {P} W")
        st.write(f"People in frame: {frame_count}")
    time.sleep(0.3)  # Sleep for 1 second before the next update
    placeholder.empty()  # Clear the placeholder for the next update


# After the loop release the cap object and destroy all windows
out.release()
cap.release()
cv2.destroyAllWindows()

