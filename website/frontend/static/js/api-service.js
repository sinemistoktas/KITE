import { state } from './state.js';
import { redrawAnnotations, resetZoom } from './canvas-tools.js';

export function getCSRFToken(name = "csrftoken") {
    const value = `; ${document.cookie}`;
    const parts = value.split(`; ${name}=`);
    return parts.length === 2 ? parts.pop().split(';').shift() : '';
}

export function processWithUNet(imageName) {
    return fetch("/process-with-unet/", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "X-CSRFToken": getCSRFToken()
        },
        body: JSON.stringify({ image_name: imageName })
    })
    .then(response => {
            if (!response.ok) {
                throw new Error("Failed to process with UNet. Status: " + response.status);
            }
            return response.json();
    });
}


export function handleAnnotations() {
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
        shapes: [{
            label: "anomaly",
            points: allPoints.map(p => [p.x, p.y]),
            color: allPoints.map(p => p.color)
        }],
        use_unet: state.unetMode
    };

    fetch("/segment/", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "X-CSRFToken": getCSRFToken()
        },
        body: JSON.stringify(payload)
    })
    .then(res => res.json())
    .then(data => {
        console.log("✅ Segmentation response:", data); // ✅ Debug output

        resetZoom();

        const resultImage = document.getElementById("segmentedResultImage");
        console.log("🖼️ resultImage element:", resultImage); // ✅ Check element
        if (!data.segmented_image) {
            console.error("❌ No segmented_image returned!");
            return;
        }

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
    });
}

export function initializeUNetPredictions(imageName){
    return processWithUNet(imageName)
        .then(data => {
            if (data.predicted_annotations && data.predicted_annotations.length > 0) {
                return data.predicted_annotations;
            }
            return [];
        })
        .catch(error => {
            console.error("Error initializing UNet predictions:", error);
            return [];
        });
}

export function handlePreprocessedImg() {
    fetch("/preprocessed-image/", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "X-CSRFToken": getCSRFToken()
        },
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
    });
}

export function handleUNetFormSubmission(event, formData) {

    const segmentationMethod = formData.get('segmentation_method');

    if (segmentationMethod === 'unet') {
        state.unetMode = true;
        return false;
    }

    state.unetMode = false;
    return false;
}