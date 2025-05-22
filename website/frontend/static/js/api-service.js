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

let segmentationMapData = null;

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
        use_unet: state.unetMode || false  // Include UNet mode if available
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

            if (data.segmentation_map) {
                segmentationMapData = data.segmentation_map;
                const showSegMapBtn = document.getElementById("showSegMapBtn");
                if (showSegMapBtn) {
                    showSegMapBtn.style.display = "inline-block";
                }
            } else {
                // Hide button if no segmentation map available
                const showSegMapBtn = document.getElementById("showSegMapBtn");
                if (showSegMapBtn) {
                    showSegMapBtn.style.display = "none";
                }
            }

            // Handle Konva stage rendering for interactive regions (from paste-2.txt)
            if (data.final_mask && data.final_mask.length > 0) {
                renderInteractiveRegions(resultImage, data.final_mask);
            }

            // Handle predicted annotations (enhanced from paste.txt)
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

                        // Handle polygon annotations (from paste.txt)
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
                                class_id: annotation.class_id || "fluid"
                            });

                            console.log("Added fluid polygon with", points.length, "points");
                        }
                        // Handle point annotations (from paste-2.txt)
                        else if (Array.isArray(annotation)) {
                            const [x, y] = Array.isArray(annotation[0]) ? annotation[0] : annotation;
                            const color = annotation[1] || "blue";

                            state.scribbles.push({
                                points: [{ x, y, color }],
                                isPrediction: true,
                                color: color
                            });
                        }
                    });

                    // Create class legend if available (from paste.txt)
                    if (data.class_info) {
                        createClassLegend(data.class_info);
                    }
                }

                redrawAnnotations();
            } catch (err) {
                console.error("Error processing annotations:", err);
            }
        })
        .catch(error => {
            console.error("Error in handleAnnotations:", error);
        });
}

export function handleSegmentationMap() {
    if (!segmentationMapData) {
        alert("No segmentation map data available.");
        return;
    }

    const popup = document.createElement("div");
    popup.style = `
        position: fixed; top: 50%; left: 50%;
        transform: translate(-50%, -50%);
        background-color: #fff; padding: 20px;
        border: 2px solid #444; z-index: 9999;
        box-shadow: 0 0 20px rgba(0,0,0,0.7);
        border-radius: 10px;
        max-width: 90vw;
        max-height: 90vh;
        overflow: auto;
    `;

    const title = document.createElement("h4");
    title.textContent = "Raw Segmentation Map";
    title.style = "margin-top: 0; margin-bottom: 15px; text-align: center;";

    const description = document.createElement("p");
    description.textContent = "This shows the raw pixel-level class predictions from the UNet model:";
    description.style = "margin-bottom: 15px; color: #666; text-align: center;";

    const legend = document.createElement("div");
    legend.style = "margin-bottom: 15px; font-size: 12px;";
    legend.innerHTML = `
        <strong>Color Legend:</strong><br>
        <span style="color: #000;">■ Black: Background</span><br>
        <span style="color: #ff0000;">■ Red: Class 1</span><br>
        <span style="color: #00ff00;">■ Green: Class 2</span><br>
        <span style="color: #0000ff;">■ Blue: Class 3</span><br>
        <span style="color: #ffff00;">■ Yellow: Class 4</span><br>
        <span style="color: #ff00ff;">■ Magenta: Class 5</span><br>
        <span style="color: #00ffff;">■ Cyan: Class 6</span><br>
        <span style="color: #800000;">■ Maroon: Class 7</span><br>
        <span style="color: #008000;">■ Dark Green: Class 8</span><br>
        <span style="color: #000080;">■ Navy: Class 9</span>
    `;

    const img = document.createElement("img");
    img.src = `data:image/png;base64,${segmentationMapData}`;
    img.style = "max-width: 100%; max-height: 70vh; display: block; margin: 0 auto;";

    const buttonContainer = document.createElement("div");
    buttonContainer.style = "text-align: center; margin-top: 15px;";

    const downloadBtn = document.createElement("a");
    downloadBtn.href = img.src;
    downloadBtn.download = "segmentation_map.png";
    downloadBtn.className = "btn btn-outline-primary me-2";
    downloadBtn.innerHTML = '<i class="fa-solid fa-download"></i> Download';

    const closeBtn = document.createElement("button");
    closeBtn.className = "btn btn-danger";
    closeBtn.innerHTML = '<i class="fa-solid fa-times"></i> Close';
    closeBtn.onclick = () => popup.remove();

    buttonContainer.append(downloadBtn, closeBtn);
    popup.append(title, description, legend, img, buttonContainer);
    document.body.appendChild(popup);
}
window.handleSegmentationMap = handleSegmentationMap;

function renderInteractiveRegions(resultImage, finalMask) {
    // Enhanced version of the Konva rendering from paste-2.txt
    resultImage.onload = () => {
        const stageContainer = document.getElementById("segmentationStage");

        if (!stageContainer) {
            console.warn("segmentationStage container not found");
            return;
        }

        const existingStage = stageContainer.querySelector(".konvajs-content");
        if (existingStage) existingStage.remove(); // clear existing Konva stage if any

        const width = resultImage.naturalWidth || resultImage.clientWidth;
        const height = resultImage.naturalHeight || resultImage.clientHeight;

        const stage = new Konva.Stage({
            container: "segmentationStage",
            width,
            height
        });

        const layer = new Konva.Layer();
        stage.add(layer);

        finalMask.forEach(({regionId, pixels, color}) => {
            const group = new Konva.Group({
                id: regionId,
                draggable: true
            });

            pixels.forEach(([y, x]) => {
                const rect = new Konva.Rect({
                    x,
                    y,
                    width: 1,
                    height: 1,
                    fill: color || "rgba(250, 37, 37, 0.98)"
                });
                group.add(rect);
            });

            group.on("click", () => {
                console.log(`Clicked region: ${regionId}`);
                // TODO: Add region deletion or modification functionality here
            });

            layer.add(group);
        });

        layer.draw();
    };
}

export function initializeUNetPredictions(imageName) {
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
        })
        .catch(error => {
            console.error("Error fetching preprocessed image:", error);
            alert("Error fetching preprocessed image. Please try again.");
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

    const resultContainer = document.getElementById("segmentationResult");
    if (resultContainer) {
        resultContainer.appendChild(legend);
    }
}