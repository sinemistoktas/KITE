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
    console.log("Segmentation response:", data);
    
    resetZoom();

    const resultImage = document.getElementById("segmentedResultImage");
    console.log("resultImage element:", resultImage);
    
    if (!data.segmented_image) {
        console.error("No segmented_image returned!");
        return;
    }

    resultImage.src = `data:image/png;base64,${data.segmented_image}`;
    document.getElementById("segmentationResult").style.display = "block";
    
    const downloadBtn = document.getElementById("downloadSegmentedImage");
    downloadBtn.href = resultImage.src;
    downloadBtn.download = "segmented_result.png";
    downloadBtn.style.display = "inline-block";

    console.log("Predicted annotations full data:", JSON.stringify(data.predicted_annotations));
    
    if (!data.predicted_annotations || data.predicted_annotations.length === 0) {
        console.log("No predicted annotations found");
        return;
    }
    
    try {
        // Clear previous predictions
        state.scribbles = state.scribbles.filter(s => !s.isPrediction);
        
        const predictedAnnotations = data.predicted_annotations;
        
        if (Array.isArray(predictedAnnotations)) {
            console.log("Annotation is an array with length:", predictedAnnotations.length);
            
            predictedAnnotations.forEach((annotation, index) => {
                console.log(`Annotation ${index} type:`, typeof annotation);
                console.log(`Annotation ${index} value:`, annotation);

                if (annotation && annotation.shape_type === "polygon" && Array.isArray(annotation.points)) {
                    const points = annotation.points.map(point => ({
                        x: point[0],
                        y: point[1],
                        color: annotation.color ? `rgb(${annotation.color[0]}, ${annotation.color[1]}, ${annotation.color[2]})` : "cyan"
                    }));

                    state.scribbles.push({
                        points: points,
                        isPrediction: true,
                        color: annotation.color ? `rgb(${annotation.color[0]}, ${annotation.color[1]}, ${annotation.color[2]})` : "cyan",
                        class_id: annotation.class_id || "fluid"  // Default to "fluid"
                    });

                    console.log("Added fluid polygon with", points.length, "points");
                }
            });
            // Create class legend if available
            if (data.class_info) {
                createClassLegend(data.class_info);
            }
        }
        
        redrawAnnotations();
    } catch (err) {
        console.error("Error processing annotations:", err);
    }
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

function createClassLegend(classInfo) {
    const existingLegend = document.getElementById("classLegend");
    if (existingLegend) {
        existingLegend.remove();
    }
    
    const legend = document.createElement("div");
    legend.id = "classLegend";
    legend.style = `
        position: absolute;
        top: 10px;
        right: 10px;
        background-color: rgba(255, 255, 255, 0.8);
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #ccc;
        max-width: 200px;
        z-index: 1000;
    `;
    
    const title = document.createElement("h4");
    title.textContent = "Classes";
    title.style = "margin-top: 0; margin-bottom: 8px;";
    legend.appendChild(title);
    
    classInfo.filter(c => c.id > 0).forEach(classItem => {
        const item = document.createElement("div");
        item.style = "display: flex; align-items: center; margin-bottom: 4px;";
        
        const colorBox = document.createElement("div");
        colorBox.style = `
            width: 15px;
            height: 15px;
            background-color: rgb(${classItem.color[0]}, ${classItem.color[1]}, ${classItem.color[2]});
            margin-right: 8px;
        `;
        
        const label = document.createElement("span");
        label.textContent = classItem.name;
        
        item.appendChild(colorBox);
        item.appendChild(label);
        legend.appendChild(item);
    });
    
    document.getElementById("segmentationResult").appendChild(legend);
}