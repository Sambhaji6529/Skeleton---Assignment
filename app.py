from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)

# Mediapipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

mp_face_mesh = mp.solutions.face_mesh.FaceMesh()  # Initialize Mediapipe Face Mesh

# Function to preprocess the frame
def preprocess_frame(frame):
    # Resize frame for faster processing
    frame = cv2.resize(frame, (640, 480))
    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb

# Function to draw face mesh
def draw_face_mesh(frame):
    frame_rgb = preprocess_frame(frame)
    results = mp_face_mesh.process(frame_rgb)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                face_landmarks, 
                mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1)
            )
    return frame

# Function to draw pose skeleton
def draw_pose_skeleton(frame):
    frame_rgb = preprocess_frame(frame)
    results = pose.process(frame_rgb)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
        )
    return frame

# Combined function to draw both face and pose
def draw_skeleton(frame):
    frame = draw_face_mesh(frame)  # Draw face mesh first
    frame = draw_pose_skeleton(frame)  # Then draw pose skeleton
    return frame

# Video feed generator
def generate_video():
    cap = cv2.VideoCapture(0)  # Open webcam

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from webcam.")
            break
        
        frame = cv2.flip(frame, 1)  # Flip the frame horizontally
        frame = draw_skeleton(frame)  # Draw skeleton on frame

        _, buffer = cv2.imencode('.jpg', frame)  # Convert frame to JPEG
        frame = buffer.tobytes()  # Convert to bytes for streaming

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
