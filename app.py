from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

# Directories
DATA_DIR = "data"
HIGH_QUALITY_DIR = "high_quality_data"

# Load pre-trained features
feature_list = pickle.load(open('artifacts/extracted_features/embedding.pkl', 'rb'))
filenames = pickle.load(open('artifacts/pickle_format_data/img_pickle_file.pkl', 'rb'))

# Initialize Flask app and SocketIO
app = Flask(__name__, static_folder='static', template_folder='templates')
socketio = SocketIO(app)

# Serve files from the 'data' directory
@app.route('/data/<path:filename>')
def serve_data(filename):
    return send_from_directory(DATA_DIR, filename)

# Serve high-quality images
@app.route('/high_quality_data/<path:filename>')
def serve_high_quality_image(filename):
    return send_from_directory(HIGH_QUALITY_DIR, filename)

# Get high-quality image for a given low-quality match
def get_high_quality_image(lookalike):
    celeb_name = os.path.basename(os.path.dirname(lookalike))
    high_quality_image_path = os.path.join(HIGH_QUALITY_DIR, celeb_name + ".jpg")
    if os.path.exists(high_quality_image_path):
        return high_quality_image_path
    else:
        return None

# Find lookalike using DeepFace and cosine similarity
def find_lookalike(image_data):
    try:
        # Convert base64 to OpenCV image
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect face
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))

        if len(faces) > 0:
            x, y, w, h = faces[0]
            cropped_face = img_rgb[y:y + h, x:x + w]
            cropped_face_resized = cv2.resize(cropped_face, (224, 224))

            # Extract features and find the most similar celebrity
            features = DeepFace.represent(cropped_face_resized, model_name='VGG-Face', enforce_detection=False)
            similarity_scores = cosine_similarity([features[0]["embedding"]], feature_list)
            index_pos = np.argmax(similarity_scores)
            lookalike = filenames[index_pos]

            # Get high-quality image URL
            high_quality_image_path = get_high_quality_image(lookalike)
            if high_quality_image_path:
                high_quality_image_url = f"/high_quality_data/{os.path.basename(high_quality_image_path)}"
            else:
                high_quality_image_url = None

            # Prepare result
            lookalike_name = os.path.basename(os.path.dirname(lookalike)).replace("_", " ")
            return {"name": lookalike_name, "image": high_quality_image_url}

        else:
            return {"error": "No face detected"}

    except Exception as e:
        return {"error": str(e)}

# WebSocket endpoint for real-time detection
@socketio.on('capture_frame')
def detect_frame(data):
    if 'image' in data:
        result = find_lookalike(data['image'])
        emit('lookalike_result', result)

# Serve the HTML front-end
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    socketio.run(app, debug=True)
