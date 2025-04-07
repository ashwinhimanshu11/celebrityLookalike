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
        nameElement.textContent = `Your Lookalike: ${data.name}`;
        lookalikeImage.src = data.image || '/static/default_image.jpg'; // Fallback for missing image
    }
});
