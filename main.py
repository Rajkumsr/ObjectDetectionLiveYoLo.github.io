import os
import torch
import cv2
from PIL import Image
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, Response, flash



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads'
app.secret_key = 'your_secret_key'

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5x')

# Global variables for live feed
cap = None
video_feed_running = False

# Function to perform object detection and update display
def detect_and_display(frame):
    # Convert frame from BGR to RGB (OpenCV uses BGR by default)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    img_pil = Image.fromarray(img)
    
    # Perform object detection
    results = model(img_pil)
    
    # Render the results on the original image
    results.render()
    
    # Convert back to BGR for OpenCV
    img_with_boxes = np.array(results.ims[0])
    img_with_boxes = cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR)
    
    return img_with_boxes

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home') 
def home():
    return render_template('index.html')

@app.route('/upload_image', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file:
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Load and display the uploaded image
            img = Image.open(file_path)
            
            # Perform object detection
            results = model(img)
            
            # Draw bounding boxes on the image
            img_with_boxes = Image.fromarray(results.render()[0])  # Convert NumPy array to PIL Image
            
            # Save the image with predicted results
            filename_processed = 'processed_' + filename
            file_path_processed = os.path.join(app.config['UPLOAD_FOLDER'], filename_processed)
            img_with_boxes.save(file_path_processed)
            
            # Prepare bounding boxes and labels for display
            boxes_labels = results.pandas().xyxy[0].to_html()
            
            return render_template('upload_image.html', filename=filename, boxes_labels=boxes_labels, filename_processed=filename_processed)
    
    return render_template('upload_image.html')

@app.route('/upload_video', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        # Handle video upload (if needed)
        pass
    
    return render_template('upload_video.html')

@app.route('/start_feed')
def start_feed():
    global cap, video_feed_running
    
    if not video_feed_running:
        # Initialize the camera
        cap = cv2.VideoCapture(0)  # Use 0 for default camera
        
        # Check if the webcam is opened correctly
        if not cap.isOpened():
            flash("Error: Could not open video stream.")
            return redirect(url_for('index'))
        
        video_feed_running = True
        
        return render_template('start_feed.html')
    else:
        flash("Video feed is already running.")
        return redirect(url_for('index'))

@app.route('/stop_feed')
def stop_feed():
    global cap, video_feed_running
    
    if video_feed_running:
        video_feed_running = False
        
        # Release the camera and close all windows
        cap.release()
        cv2.destroyAllWindows()
        
        return render_template('index.html')
    else:
        flash("No video feed is currently running.")
        return redirect(url_for('index'))

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    global cap
    
    while video_feed_running:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Perform object detection and update display
        img_with_boxes = detect_and_display(frame)
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', img_with_boxes)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

if __name__ == '__main__':
    app.run(debug=True)
