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

        // Start capturing frames every 8 seconds
        setInterval(() => {
            captureFrameAndSend();
        }, 8000);
    })
    .catch((err) => console.error("Error accessing camera: ", err));

// Function to capture frame and send to server
function captureFrameAndSend() {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const imageData = canvas.toDataURL('image/jpeg').split(',')[1];

    // Send the captured frame to the server
    socket.emit('capture_frame', { image: imageData });
}

// Handle the lookalike result from the server
let animationTimeout;

socket.on('lookalike_result', (data) => {
    if (data.error) {
        nameElement.textContent = data.error;
        clearImage(); // Clear the image and text if there's an error
    } else {
        nameElement.textContent = `Your Lookalike: ${data.name}`;
        displayWithAnimation(data.image); // Display the result with animation
    }
});

// Function to display the image with animation and hold it
function displayWithAnimation(imageSrc) {
    clearTimeout(animationTimeout); // Clear any previous timeout to avoid overlap

    lookalikeImage.style.animation = "none"; // Reset animation
    lookalikeImage.offsetHeight; // Trigger reflow to apply the reset
    lookalikeImage.src = imageSrc; // Set the new image
    lookalikeImage.style.animation = "slide-in 0.5s ease forwards"; // Start animation

    // Hold the image for 8 seconds, then clear it
    animationTimeout = setTimeout(() => {
        clearImage();
    }, 8000);
}

// Function to clear the image and text
function clearImage() {
    lookalikeImage.style.animation = "none"; // Reset animation
    lookalikeImage.src = ""; // Clear the image
    nameElement.textContent = ""; // Clear the name text
}
