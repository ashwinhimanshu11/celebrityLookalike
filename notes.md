## app.py

```python
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
```

## 1_generate_img_pkl.py

```python
from src.utils.all_utils import read_yaml, create_directory
import argparse
import os
import pickle
import logging
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging_str = "[%(asctime)s:%(levelname)s: %(module)s: %(message)s]"
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'running_log.log'),
                    level=logging.INFO, format=logging_str, filemode='a')


def generate_data_pickle_file(config_path, params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    artifacts = config['artifacts']
    artifacts_dir = artifacts['artifacts_dir']
    pickle_format_data_dir = artifacts['pickle_format_data_dir']
    img_pickle_file_name = artifacts['img_pickle_file_name']
    raw_local_dir_path = os.path.join(artifacts_dir, pickle_format_data_dir)
    create_directory(dirs=[raw_local_dir_path])

    pickle_file = os.path.join(raw_local_dir_path, img_pickle_file_name)
    data_path = params['base']['data_path']
    actors = os.listdir(data_path)
    filenames = []

    for actor in actors:
        for file in os.listdir(os.path.join(data_path, actor)):
            filenames.append(os.path.join(data_path, actor, file))

    logging.info(f"Total actors are: {len(actors)}")
    logging.info(f"Total actors images are: {len(filenames)}")

    pickle.dump(filenames, open(pickle_file, 'wb'))


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--config', "-c", default='config/config.yaml')
    args.add_argument('--params', "-p", default='params.yaml')
    parsed_args = args.parse_args()

    try:
        logging.info(">>>>> stage_01 is started")
        generate_data_pickle_file(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info("stage_01 is completed")
    except Exception as e:
        logging.exception(e)
        raise e
```

## 2_feature_extractor.py

```python
from deepface import DeepFace
import os
import pickle
from tqdm import tqdm
from src.utils.all_utils import read_yaml, create_directory

def feature_extractor(config_path, params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    artifacts = config['artifacts']
    artifacts_dir = artifacts['artifacts_dir']
    pickle_format_data_dir = artifacts['pickle_format_data_dir']
    img_pickle_file_name = artifacts['img_pickle_file_name']
    img_pickle_file_name = os.path.join(artifacts_dir, pickle_format_data_dir, img_pickle_file_name)
    filenames = pickle.load(open(img_pickle_file_name, 'rb'))

    # Directory for extracted features
    feature_extraction_path = os.path.join(artifacts_dir, artifacts['feature_extraction_dir'])
    create_directory(dirs=[feature_extraction_path])

    feature_name = os.path.join(feature_extraction_path, artifacts['extracted_features_name'])
    features = []

    # Use DeepFace for feature extraction
    for file in tqdm(filenames):
        try:
            embeddings = DeepFace.represent(img_path=file, model_name='VGG-Face', enforce_detection=False)
            if embeddings:
                features.append(embeddings[0]["embedding"])
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue

    # Save extracted features
    pickle.dump(features, open(feature_name, 'wb'))
    print(f"Features saved to {feature_name}")

if __name__ == "__main__":
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument('--config', "-c", default='config/config.yaml')
    args.add_argument('--params', "-p", default='params.yaml')
    parsed_args = args.parse_args()
    feature_extractor(config_path=parsed_args.config, params_path=parsed_args.params)
    
```

## index.html

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Celebrity Lookalike Finder</title>
    <link rel="stylesheet" href="/static/styles.css">
    <script src="https://cdn.socket.io/4.0.1/socket.io.min.js"></script>
</head>
<body>
    <div class="container">
        <h1>Celebrity Lookalike Finder</h1>
        <div class="content">
            <div class="live-feed">
                <video id="video" autoplay></video>
            </div>
            <div class="lookalike">
                <h2 id="name">Your Lookalike:</h2>
                <img id="lookalike" alt="Lookalike">
            </div>
        </div>
    </div>
    <script src="/static/scripts.js"></script>
</body>
</html>
```

## styles.css

```css
body {
    font-family: Arial, sans-serif;
    background: linear-gradient(135deg, #6a11cb, #2575fc);
    color: white;
    text-align: center;
    margin: 0;
    padding: 0;
}

.container {
    max-width: 1200px;
    margin: 20px auto;
    padding: 20px;
    background: rgba(0, 0, 0, 0.7);
    border-radius: 10px;
}

h1 {
    margin-bottom: 20px;
    font-size: 36px;
    color: #fff;
}

.content {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 20px;
}

.live-feed video {
    width: 600px;
    max-height: 400px;
    border: 3px solid #fff;
    border-radius: 10px;
}

.lookalike {
    text-align: center;
    flex: 1;
    background: rgba(255, 255, 255, 0.1);
    padding: 20px;
    border-radius: 10px;
}

#name {
    font-size: 24px;
    margin-bottom: 20px;
    color: #FFD700;
}

#lookalike {
    width: 300px;
    height: 400px;
    object-fit: cover;
    border-radius: 10px;
    border: 3px solid #fff;
    background: rgba(255, 255, 255, 0.2);
}
```

## scripts.js

```js
const video = document.getElementById('video');
const nameElement = document.getElementById('name');
const lookalikeImage = document.getElementById('lookalike');

// Connect to Socket.IO
const socket = io.connect('http://127.0.0.1:5000');

// Start video feed
navigator.mediaDevices
    .getUserMedia({ video: true })
    .then((stream) => {
        video.srcObject = stream;

        // Send frames periodically
        setInterval(() => {
            captureFrameAndSend();
        }, 5000);
    })
    .catch((err) => console.error("Error accessing camera: ", err));

// Function to capture frame and send to the server
function captureFrameAndSend() {
    const canvas = document.createElement('canvas'); // Temporary canvas
    const ctx = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const imageData = canvas.toDataURL('image/jpeg').split(',')[1];
    socket.emit('capture_frame', { image: imageData });
}

// Handle the lookalike result from the server
socket.on('lookalike_result', (data) => {
    if (data.error) {
        nameElement.textContent = data.error;
        lookalikeImage.src = ''; // Clear the image on error
    } else {
        nameElement.textContent = Your Lookalike: ${data.name};
        lookalikeImage.src = data.image || '/static/default_image.jpg'; // Fallback for missing image
    }
});
```