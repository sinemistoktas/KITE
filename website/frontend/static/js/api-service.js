import { state } from './state.js';
import { redrawAnnotations, resetZoom } from './canvas-tools.js';

export function getCSRFToken(name = "csrftoken") {
    // First try to get token from window object
    if (window.csrfToken) {
        console.log('Using CSRF token from window object');
        return window.csrfToken;
    }
    
    // Fallback to cookie
    const value = `; ${document.cookie}`;
    const parts = value.split(`; ${name}=`);
    if (parts.length === 2) {
        const token = parts.pop().split(';').shift();
        console.log('Using CSRF token from cookie');
        return token;
    }
    // else, no CSRF token found
    console.warn('No CSRF token found in window object or cookies');
    return '';
}

function fetchWithCSRF(url, options = {}) {
    const csrfToken = getCSRFToken();
    if (!csrfToken) {
        console.error('No CSRF token available');
        return Promise.reject(new Error('No CSRF token available'));
    }

    const defaultOptions = {
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': csrfToken
        },
        credentials: 'same-origin'  // Added this to ensure cookies are sent
    };

    return fetch(url, { ...defaultOptions, ...options });
}

export function handleAnnotations() {
    // Get the selected algorithm or default to 'kite'
    const algorithm = document.getElementById('algorithm')?.value || 'kite'; // might need to change this later
    console.log('Using algorithm:', algorithm);

    state.scribbles = state.scribbles.filter(s => !s.isPrediction);
    redrawAnnotations();

    const allPoints = state.scribbles.flatMap(s =>
        s.points.map(p => ({
            x: p.x,
            y: p.y,
            color: s.color
        }))
    );

    const payload = {
        image_name: state.imageName,
        algorithm: algorithm, // Add the selected algorithm
        shapes: [{
            label: "anomaly",
            points: allPoints.map(p => [p.x, p.y]),
            color: allPoints.map(p => p.color)
        }]
    };

    fetchWithCSRF("/segment/", {
        method: "POST",
        body: JSON.stringify(payload)
    })
    .then(res => {
        console.log('Response status:', res.status);
        if (!res.ok) {
            return res.text().then(text => {
                throw new Error(`HTTP error! status: ${res.status}, message: ${text}`);
            });
        }
        return res.json();
    })
    .then(data => {
        console.log('Received data:', data);
        resetZoom();

        const resultImage = document.getElementById("segmentedResultImage");
        resultImage.src = `data:image/png;base64,${data.segmented_image}`;

        document.getElementById("segmentationResult").style.display = "block";
        const downloadBtn = document.getElementById("downloadSegmentedImage");
        downloadBtn.href = resultImage.src;
        downloadBtn.download = "segmented_result.png";
        downloadBtn.style.display = "inline-block";

        const predictedPoints = data.predicted_annotations || [];
        const strokes = predictedPoints.map(p => {
            if (Array.isArray(p[0])) {
                const [[x, y], color] = p;
                return { x, y, color };
            } else {
                const [x, y] = p;
                return { x, y, color: "blue" };
            }
        });

        state.scribbles.push({
            points: strokes,
            isPrediction: true,
            color: "blue"
        });

        redrawAnnotations();
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error processing segmentation: ' + error.message);
    });
}

export function handlePreprocessedImg() {
    fetchWithCSRF("/preprocessed-image/", {
        method: "POST",
        body: JSON.stringify({ image_name: state.imageName })
    })
    .then(res => res.json())
    .then(data => {
        if (!data?.preprocessed_image) return alert("No preprocessed image returned.");

        const popup = document.createElement("div");
        popup.style = `
            position: fixed; top: 50%; left: 50%;
            transform: translate(-50%, -50%);
            background-color: #fff; padding: 10px;
            border: 2px solid #444; z-index: 9999;
            box-shadow: 0 0 10px rgba(0,0,0,0.5)
        `;

        const img = document.createElement("img");
        img.src = `data:image/png;base64,${data.preprocessed_image}`;
        img.style.maxWidth = "100%";
        img.style.maxHeight = "80vh";

        const close = document.createElement("button");
        close.className = "btn btn-danger mt-3";
        close.innerText = "Close";
        close.onclick = () => popup.remove();

        popup.append(img, close);
        document.body.appendChild(popup);
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error processing preprocessed image: ' + error.message);
    });
}