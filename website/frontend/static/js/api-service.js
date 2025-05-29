import { state } from './state.js';
import { redrawAnnotations, resetZoom } from './canvas-tools.js';

export function getCSRFToken(name = "csrftoken") {
    // First try to get token from window object (improved from dev branch)
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
        credentials: 'same-origin'
    };

    return fetch(url, { ...defaultOptions, ...options });
}

export function processWithUNet(imageName) {
    return fetchWithCSRF("/process-with-unet/", {
        method: "POST",
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
    // Remove existing mask preview panel
    const existingPanel = document.getElementById('maskPreviewPanel');
    if (existingPanel) {
        existingPanel.remove();
    }

    // Get the selected algorithm or method
    const algorithmRadio = document.querySelector('input[name="algorithm"]:checked');
    const algorithm = algorithmRadio?.value || window.algorithm || 'kite';
    const segmentationMethod = state.segmentationMethod || 'traditional';
    
    console.log('Using algorithm:', algorithm);
    console.log('Using segmentation method:', segmentationMethod);

    if (algorithm === 'unet') {
        console.log("UNet selected, using processWithUNet()");
        return processWithUNet(state.imageName)
            .then(data => {
                console.log("UNet segmentation response:", data);
                resetZoom();

                window.lastSegmentationData = data;

                const resultImage = document.getElementById("segmentedResultImage");
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
                    if (showSegMapBtn) showSegMapBtn.style.display = "inline-block";
                }

                const stageContainer = document.getElementById("segmentationStage");
                if (stageContainer) {
                    stageContainer.innerHTML = "";
                    stageContainer.style.width = resultImage.naturalWidth + "px";
                    stageContainer.style.height = resultImage.naturalHeight + "px";
                }

                if (data.final_mask && data.final_mask.length > 0 && stageContainer) {
                    renderInteractiveRegions(resultImage, data.final_mask,
                        resultImage.naturalWidth || resultImage.clientWidth,
                        resultImage.naturalHeight || resultImage.clientHeight,
                        stageContainer);
                }

                if (data.predicted_annotations && data.predicted_annotations.length > 0) {
                    try {
                        const predictedAnnotations = data.predicted_annotations;
                        state.scribbles = state.scribbles.filter(s => !s.isPrediction);

                        predictedAnnotations.forEach(annotation => {
                            if (annotation.shape_type === "polygon") {
                                const points = annotation.points.map(p => ({ x: p[0], y: p[1], color: "cyan" }));
                                state.scribbles.push({
                                    points: points,
                                    isPrediction: true,
                                    color: "cyan",
                                    class_id: annotation.class_id || "fluid"
                                });
                            }
                        });

                        if (data.class_info) createClassLegend(data.class_info);
                        redrawAnnotations();
                    } catch (err) {
                        console.error("Error parsing UNet annotations:", err);
                    }
                }
            })
            .catch(err => {
                console.error("UNet segmentation failed:", err);
                alert("UNet segmentation failed. Check the console for more details.");
            });
    }


    const originalScribbles = state.scribbles;
    state.scribbles = state.scribbles.filter(s => !s.isPrediction);

    const allPoints = state.scribbles.flatMap(s =>
        s.points.map(p => ({
            x: p.x,
            y: p.y,
            color: s.color,
            layerId: s.layerId  // Include layer ID
        }))
    );

    const payload = {
        image_name: state.imageName,
        algorithm: algorithm, // Add the selected algorithm
        segmentation_method: segmentationMethod, // Add segmentation method for legacy support
        use_unet: state.unetMode || false, // Include UNet mode
        shapes: [{
            label: "anomaly",
            points: allPoints.map(p => [p.x, p.y]),
            color: allPoints.map(p => p.color),
            layerId: allPoints.map(p => p.layerId)  // Send layer IDs
        }]
    };

    return fetchWithCSRF("/segment/", {
        method: "POST",
        body: JSON.stringify(payload)
    })
    .then(res => {
        if (!res.ok) {
            return res.text().then(text => {
                throw new Error(`HTTP error! status: ${res.status}, message: ${text}`);
            });
        }
        return res.json();
    })
    .then(data => {
        console.log("Segmentation response:", data);
        resetZoom();
        window.lastSegmentationData = data;

        console.log(data.segmentation_mask_npy);
        console.log("Segment response:", data);

        const resultImage = document.getElementById("segmentedResultImage");
        console.log("resultImage element:", resultImage);

        if (data.segmented_image) {
            resultImage.src = `data:image/png;base64,${data.segmented_image}`;
            document.getElementById("segmentationResult").style.display = "block";
            
            const downloadBtn = document.getElementById("downloadSegmentedImage");
            downloadBtn.href = resultImage.src;
            downloadBtn.download = "segmented_result.png";
            downloadBtn.style.display = "inline-block";
        }


        if (!data.segmented_image) {
            console.error("No segmented_image returned!");
            return;
        }
        // Handle segmentation map display (for UNet)
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

     

        // Create mask preview panel if individual masks are available
        if (data.segmentation_masks && data.segmentation_masks.individual_masks && data.segmentation_masks.individual_masks.length > 0) {
            createMaskPreviewPanel(data.segmentation_masks);
        }



        // Handle predicted annotations (enhanced from both versions)
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

                // Handle different annotation formats
                const processedPoints = [];

                 predictedAnnotations.forEach((annotation, index) => {

                    // Handle polygon annotations (from UNet/advanced algorithms)
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
                    // Handle point annotations (from KITE/traditional algorithms)
                    else if (Array.isArray(annotation)) {
                        if (Array.isArray(annotation[0])) {
                            // Format: [[x, y], color]
                            const [[x, y], color] = annotation;           
                            
                            let scaledX, scaledY;
                            // Scale fix to ensure that the coordinates are NOT scaled again!
                            if (state.coordinatesAlreadyScaled) {
                                scaledX = x * state.displayScale;
                                scaledY = y * state.displayScale;
                            } else {
                                const scaleX = state.originalImageDimensions.width / 512;  
                                const scaleY = state.originalImageDimensions.height / 224; 
                                scaledX = x * scaleX;
                                scaledY = y * scaleY;
                            }
                            
                            processedPoints.push({ 
                                x: scaledX, 
                                y: scaledY, 
                                color: color || "blue"
                            });
                        } else {
                            // Format: [x, y]
                            const [x, y] = annotation;
                            
                            let scaledX, scaledY;
                            if (state.coordinatesAlreadyScaled) {
                                scaledX = x * state.displayScale;
                                scaledY = y * state.displayScale;
                            } else {
                                const scaleX = state.originalImageDimensions.width / 512;  
                                const scaleY = state.originalImageDimensions.height / 224; 
                                scaledX = x * scaleX;
                                scaledY = y * scaleY;
                            }
                            
                            processedPoints.push({ 
                                x: scaledX, 
                                y: scaledY, 
                                color: "blue" 
                            });
                        }
                    }
                });

                // Add processed points as a single scribble if any exist
                if (processedPoints.length > 0) {
                    state.scribbles.push({
                        points: processedPoints,
                        isPrediction: true,
                        color: "blue"
                    });
                }

                // Create class legend if available
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
        alert('Error processing segmentation: ' + error.message);
    });
}

// previously were used for interavity with konva, now it is unused 
// delete this and its references later 
function renderInteractiveRegions(resultImage, finalMask, width, height, stageContainer) {
    // Enhanced version combining both approaches
    resultImage.onload = () => {
        if (!stageContainer) {
            console.warn("segmentationStage container not found");
            return;
        }

        const existingStage = stageContainer.querySelector(".konvajs-content");
        if (existingStage) existingStage.remove(); // clear existing Konva stage if any

        const stage = new Konva.Stage({
            container: "segmentationStage",
            width: width,
            height: height
        });

        const layer = new Konva.Layer();
        stage.add(layer);

        // Current problems noted:
        // 1. the deleted regions are not removed from the canvas since it takes from the backend, it needs to be refreshed every time
        // 2. it works a bit slow when there are too many regions on the image, try other alternatives rather than grouping them all
        // 3. canvas tools and events should be updated accordingly to support this function
        finalMask.forEach(({regionId, pixels, color}) => {
            const group = new Konva.Group({
                id: regionId,
                draggable: true
            });

            pixels.forEach(([y, x]) => {
                const rect = new Konva.Rect({
                    x: x * state.displayScale,
                    y: y * state.displayScale,
                    width: 1,
                    height: 1,
                    fill: color || "rgba(233, 37, 37, 0.98)"
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

// Function to create the "segmentation masks" preview panel
function createMaskPreviewPanel(maskData) {
    const existingPanel = document.getElementById('maskPreviewPanel');
    if (existingPanel) existingPanel.remove();

    if (!maskData.individual_masks || maskData.individual_masks.length === 0) {
        return;
    }
    
    const maskPanel = document.createElement('div');
    maskPanel.id = 'maskPreviewPanel';
    maskPanel.className = 'mask-preview-panel';

    // Insert next to segmentation result.
    const flexContainer = document.querySelector('#segmentationResult .d-flex.flex-row');
    if (flexContainer) {
        flexContainer.appendChild(maskPanel);
    }

    maskPanel.innerHTML = `
        <div class="mask-panel-header">
            <h5 class="mask-panel-title">
                <i class="fa-solid fa-layer-group me-2"></i>
                Segmentation Masks
            </h5>
            <span class="mask-count">${maskData.individual_masks.length} masks detected</span>
        </div>
        
        <div class="combined-mask-preview">
            <div class="combined-mask-container">
                <img src="${maskData.combined_mask_url}" alt="Combined Masks" class="combined-mask-image" id="combinedMaskImage">
                <div class="mask-selection-overlay" id="maskSelectionOverlay"></div>
            </div>
            <p class="mask-instruction">Select multiple masks for bulk download</p>
        </div>
        
        <!-- Selection Controls -->
        <div class="selection-controls">
            <div class="select-all-container">
                <label class="select-toggle">
                    <input type="checkbox" id="selectAllMasks">
                    <span class="toggle-slider"></span>
                    <span class="toggle-label">Select All</span>
                </label>
            </div>
            <button class="bulk-download-btn" id="bulkDownloadBtn" disabled>
                <i class="fa-solid fa-download"></i>
                Download Selected (<span id="selectedCount">0</span>)
            </button>
        </div>
        
        <div class="mask-list" id="maskList"></div>
    `;

    // Create individual mask items.
    const maskList = document.getElementById('maskList');
    maskData.individual_masks.forEach((mask, index) => {
        const maskItem = document.createElement('div');
        maskItem.className = 'mask-list-item';
        maskItem.dataset.maskIndex = index;

        maskItem.innerHTML = `
            <div class="mask-item-left">
                <label class="select-toggle">
                    <input type="checkbox" class="mask-select-checkbox" data-npy-url="${mask.npyUrl}" data-filename="${mask.npyFilename}">
                    <span class="toggle-slider"></span>
                </label>
                <div class="mask-item-info">
                    <div class="mask-color-indicator" style="background-color: ${mask.color}"></div>
                    <div class="mask-details">
                        <span class="mask-name">Region ${index + 1}</span>
                        <span class="mask-id">${mask.regionId}</span>
                    </div>
                </div>
            </div>
            <div class="mask-download-buttons">
                <button class="btn btn-sm btn-outline-primary download-btn" data-url="${mask.npyUrl}" data-filename="${mask.npyFilename}" data-type="npy">
                    <i class="fa-solid fa-database"></i> NPY
                </button>
                <button class="btn btn-sm btn-outline-success download-btn" data-url="${mask.pngUrl}" data-filename="${mask.pngFilename}" data-type="png">
                    <i class="fa-solid fa-image"></i> PNG
                </button>
            </div>
        `;

        maskList.appendChild(maskItem);
    });

    setupMaskSelectionEvents(maskPanel, maskData);
    maskPanel.style.display = 'block';
}

// Function for handling selection events and bulk download
function setupMaskSelectionEvents(maskPanel, maskData) {
    const selectAllCheckbox = maskPanel.querySelector('#selectAllMasks');
    const bulkDownloadBtn = maskPanel.querySelector('#bulkDownloadBtn');
    const selectedCountSpan = maskPanel.querySelector('#selectedCount');
    const maskCheckboxes = maskPanel.querySelectorAll('.mask-select-checkbox');

    function updateSelectionState() {
        const selectedMasks = maskPanel.querySelectorAll('.mask-select-checkbox:checked');
        const selectedCount = selectedMasks.length;

        selectedCountSpan.textContent = selectedCount;
        bulkDownloadBtn.disabled = selectedCount === 0;

        if (selectedCount === 0) {
            selectAllCheckbox.indeterminate = false;
            selectAllCheckbox.checked = false;
        } else if (selectedCount === maskCheckboxes.length) {
            selectAllCheckbox.indeterminate = false;
            selectAllCheckbox.checked = true;
        } else {
            selectAllCheckbox.indeterminate = true;
        }
    }

    selectAllCheckbox.addEventListener('change', () => {
        const shouldSelect = selectAllCheckbox.checked;
        maskCheckboxes.forEach(checkbox => {
            checkbox.checked = shouldSelect;
        });
        updateSelectionState();
    });

    maskCheckboxes.forEach(checkbox => {
        checkbox.addEventListener('change', updateSelectionState);
    });

    bulkDownloadBtn.addEventListener('click', async () => {
        const selectedCheckboxes = maskPanel.querySelectorAll('.mask-select-checkbox:checked');

        if (selectedCheckboxes.length === 0) return;

        const originalText = bulkDownloadBtn.innerHTML;
        bulkDownloadBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Creating Combined Mask...';
        bulkDownloadBtn.disabled = true;

        try {
            const selectedMasks = [];
            selectedCheckboxes.forEach((checkbox) => {
                const maskIndex = parseInt(checkbox.closest('.mask-list-item').dataset.maskIndex);
                const maskInfo = maskData.individual_masks[maskIndex];

                const finalMaskData = window.lastSegmentationData?.final_mask?.find(m => m.regionId === maskInfo.regionId);

                if (finalMaskData) {
                    selectedMasks.push({
                        regionId: maskInfo.regionId,
                        pixels: finalMaskData.pixels,
                        color: maskInfo.color
                    });
                }
            });

            const response = await fetchWithCSRF("/bulk-download-masks/", {
                method: "POST",
                body: JSON.stringify({
                    selected_masks: selectedMasks,
                    image_name: state.imageName
                })
            });

            const result = await response.json();

            if (result.download_url) {
                // Download the combined file.
                const link = document.createElement('a');
                link.href = result.download_url;
                link.download = result.filename;
                link.click();
            } else {
                alert('Error creating combined mask: ' + (result.error || 'Unknown error'));
            }

        } catch (error) {
            console.error('Bulk download error:', error);
            alert('Error downloading masks. Please try again.');
        }

        setTimeout(() => {
            bulkDownloadBtn.innerHTML = originalText;
            updateSelectionState();
        }, 1000);
    });

    maskPanel.addEventListener('click', (e) => {
        if (e.target.closest('.download-btn')) {
            const button = e.target.closest('.download-btn');
            const url = button.dataset.url;
            const filename = button.dataset.filename;

            const link = document.createElement('a');
            link.href = url;
            link.download = filename;
            link.click();

            // Visual feedback
            button.style.background = '#28a745';
            button.style.color = 'white';
            setTimeout(() => {
                button.style.background = '';
                button.style.color = '';
            }, 1000);
        }
    });

    updateSelectionState();
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
            box-shadow: 0 0 10px rgba(0,0,0,0.5);
            border-radius: 10px;
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

export function loadAnnotations() {
    const fileInput = document.getElementById('annotationFileInput');
    fileInput.click();
    
    fileInput.onchange = function(event) {
        const file = event.target.files[0];
        if (!file) return;
        
        if (!file.name.endsWith('.npy')) {
            alert('Please select a .npy file');
            return;
        }
        
        const formData = new FormData();
        formData.append('annotation_file', file);
        formData.append('image_name', state.imageName);
        
        const csrfToken = getCSRFToken();
        
        fetch('/load-annotations/', {
            method: 'POST',
            headers: {
                'X-CSRFToken': csrfToken
            },
            credentials: 'same-origin',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success && data.annotations) {
                displayLoadedAnnotations(data.annotations);
            } else {
                alert('Error loading annotations: ' + (data.error || 'Unknown error'));
            }
        })
        .catch(error => {
            console.error('Error loading annotations:', error);
            alert('Error loading annotations: ' + error.message);
        });
    };
}

function displayLoadedAnnotations(annotations) {
    console.log('Loading annotations:', annotations);
    
    // FOR DEBUGGING!!
    const img = document.getElementById("uploadedImage");
    const canvas = state.annotationCanvas;
    
    // Clear existing annotations first.
    state.scribbles = state.scribbles.filter(s => s.isPrediction);
    
    const layersContainer = document.getElementById('layersContainer');
    if (layersContainer) {
        while (layersContainer.firstChild) {
            layersContainer.removeChild(layersContainer.firstChild);
        }
    }
    state.visibleLayerIds = [];
    state.layerCounter = 0;
    
    import('./layers.js').then(module => {
        const { createLayer } = module;
        
        // Create layers and annotations for each loaded region.
        annotations.forEach((annotation, index) => {
            const layerName = `Loaded Layer ${annotation.value}`;
            const layerId = createLayer(layerName, annotation.color);
            console.log(`Processing annotation ${annotation.value} with ${annotation.pixels.length} pixels`);
            if (annotation.pixels.length > 0) {
                const rawXs = annotation.pixels.map(p => p[1]);
                const rawYs = annotation.pixels.map(p => p[0]);
            }
            
            const points = annotation.pixels.map(pixel => {
                let x = pixel[1];
                let y = pixel[0];

                
                return { x, y };
            });
            
            
            // Add each pixel as an individual dot annotation.
            points.forEach(point => {
                state.scribbles.push({
                    points: [point],
                    isPrediction: false,
                    color: annotation.color,
                    layerId: layerId,
                    isLoadedAnnotation: true,
                    isDot: true
                });
            });
            
            console.log(`Added ${points.length} dots for region ${annotation.value}`);
        });
        
        import('./canvas-tools.js').then(module => {
            const { redrawAnnotations } = module;
            redrawAnnotations();
        });
        
        console.log(`Loaded ${annotations.length} annotation regions`);
    });
}
export function downloadAnnotations() {
    const userAnnotations = state.scribbles.filter(s => !s.isPrediction);
    
    if (userAnnotations.length === 0) {
        alert('No annotations to download. Please add some annotations first.');
        return;
    }
    
    const layerGroups = {};
    const layerOrder = {};
    
    userAnnotations.forEach(stroke => {
        const layerId = stroke.layerId || 'default';
        
        if (!layerGroups[layerId]) {
            layerGroups[layerId] = [];
            const layerElement = document.getElementById(layerId);
            const layerName = layerElement ? 
                layerElement.querySelector('input[type="text"]').value : 
                'Unknown Layer';
            layerOrder[layerId] = {
                name: layerName,
                order: Object.keys(layerGroups).length
            };
        }
        
        // Determine stroke type!!!! This is important for the backend to find the perfect array.
        let strokeType = 'line';

        if (stroke.isDot || stroke.points.length === 1) {
            strokeType = 'dot';
        } else if (stroke.isBox) {
            strokeType = 'box';
        } else if (stroke.isFilled || stroke.type === 'fill') {
            strokeType = 'fill';
        } else if (stroke.isLoadedAnnotation) {
            strokeType = 'dot';
        } else if (stroke.layerId) {
            // Check layer type from the DOM to determine if it's a fill.
            const layerElement = document.getElementById(stroke.layerId);
            if (layerElement) {
                const layerType = layerElement.getAttribute('data-layer-type');
                if (layerType && layerType.toLowerCase().includes('fill')) {
                    strokeType = 'fill';
                }
            }
        }
        
        // Add stroke with its TYPE and points.
        layerGroups[layerId].push({
            type: strokeType,
            points: stroke.points.map(point => ({
                x: point.x,
                y: point.y
            }))
        });
    });
    
    const annotationsData = Object.keys(layerGroups).map((layerId, index) => ({
        layer_id: layerId,
        layer_name: layerOrder[layerId].name,
        layer_order: layerOrder[layerId].order,
        strokes: layerGroups[layerId]
    }));
    
    console.log('Preparing to download annotations with stroke types:', annotationsData);
    
    const payload = {
        annotations: annotationsData,
        image_dimensions: state.originalImageDimensions,
        image_name: state.imageName
    };
    
    const downloadBtn = document.getElementById('downloadAnnotationsBtn');
    const originalText = downloadBtn.innerHTML;
    downloadBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Creating Annotation File...';
    downloadBtn.disabled = true;
    
    fetchWithCSRF("/download-annotations/", {
        method: "POST",
        body: JSON.stringify(payload)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        console.log('Download response:', data);
        
        if (data.download_url) {
            const link = document.createElement('a');
            link.href = data.download_url;
            link.download = data.filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            
            downloadBtn.innerHTML = '<i class="fa-solid fa-check"></i> Downloaded!';
            downloadBtn.style.backgroundColor = '#28a745';
            downloadBtn.style.borderColor = '#28a745';
            
            const layerCount = Object.keys(layerGroups).length;
            const totalPoints = Object.values(layerGroups).reduce((sum, points) => sum + points.length, 0);
            
        } else {
            throw new Error(data.error || 'Unknown error occurred');
        }
    })
    .catch(error => {
        console.error('Error downloading annotations:', error);
        alert('Error downloading annotations: ' + error.message);
        
        downloadBtn.innerHTML = '<i class="fa-solid fa-exclamation-triangle"></i> Error';
        downloadBtn.style.backgroundColor = '#dc3545';
        downloadBtn.style.borderColor = '#dc3545';
    })
    .finally(() => {
        setTimeout(() => {
            downloadBtn.innerHTML = originalText;
            downloadBtn.disabled = false;
            downloadBtn.style.backgroundColor = '';
            downloadBtn.style.borderColor = '';
        }, 3000);
    });
}

export function loadSegmentationResultsAsAnnotations(segmentationData) {
    if (!segmentationData || !segmentationData.final_mask) {
        console.log('No segmentation results to load as annotations');
        return;
    }

    const img = document.getElementById("uploadedImage");
    const canvas = state.annotationCanvas;

    console.log('Hiding contour predictions for editing mode');
    state.scribbles = state.scribbles.filter(s => !s.isPrediction);
    state.isEditingSegmentationResults = true;
    state.showPredictions = false; // Explicitly hide predictions


    state.scribbles = state.scribbles.filter(s => !s.isSegmentationResult);
   
    const layersContainer = document.getElementById('layersContainer');
    
    import('./layers.js').then(module => {
        const { createLayer } = module;
        
        segmentationData.final_mask.forEach((maskData, index) => {
            const regionId = maskData.regionId;
            const pixels = maskData.pixels;
            const color = maskData.color || "#ff0000";
            
            
            const layerName = `Segmented ${regionId}`;
            const layerId = createLayer(layerName, color);
            
            const points = pixels.map(pixel => {
                let x = pixel[1];
                let y = pixel[0]; 

                x = x * state.displayScale;
                y = y * state.displayScale;
            
                return { x, y };
            });
            
            if (points.length > 0) {
                const pointXs = points.map(p => p.x);
                const pointYs = points.map(p => p.y);
            }
            
            points.forEach(point => {
                state.scribbles.push({
                    points: [point], 
                    isPrediction: false,
                    color: color,
                    layerId: layerId,
                    isLoadedAnnotation: false, 
                    isSegmentationResult: true, 
                    isDot: true 
                });
            });
            
        });

        showEditingModeIndicator();
        import('./canvas-tools.js').then(module => {
            const { redrawAnnotations } = module;
            redrawAnnotations();
        });
        
        console.log(`Successfully loaded ${segmentationData.final_mask.length} segmentation results as editable annotations!`);
    });
}

function showEditingModeIndicator(autoMode= false) {
    // Update the "Edit Results on Canvas" button to show active state
    const loadBtn = document.getElementById('loadResultsAsAnnotationsBtn');
    if (loadBtn) {
        loadBtn.style.display = 'none'; // Hide the button for now can be deleted directly later
    }
}

window.exitEditingMode = function() {
    console.log('Exiting editing mode - restoring contours');
    
    state.isEditingSegmentationResults = false;
    state.showPredictions = true; // Re-enable predictions
    
    const banner = document.getElementById('editingModeBanner');
    if (banner) banner.remove();
    
    
    
    import('./canvas-tools.js').then(module => {
        const { redrawAnnotations } = module;
        redrawAnnotations();
    });

    };

export function handleAnnotationsWithResultLoading() {
    



    const algorithm = document.getElementById('algorithm')?.value || window.algorithm || 'kite';
    const segmentationMethod = state.segmentationMethod || 'traditional';
    
    console.log('Using algorithm:', algorithm);
    console.log('Using segmentation method:', segmentationMethod);

    if (!state.isEditingSegmentationResults) {
        state.scribbles = state.scribbles.filter(s => !s.isPrediction);
    } else {
        state.scribbles = state.scribbles.filter(s => !s.isPrediction);
    }

    const allPoints = state.scribbles.flatMap(s =>
        s.points.map(p => ({
            x: p.x,
            y: p.y,
            color: s.color,
            layerId: s.layerId
        }))
    );

    const payload = {
        image_name: state.imageName,
        algorithm: algorithm,
        segmentation_method: segmentationMethod,
        use_unet: state.unetMode || false,
        shapes: [{
            label: "anomaly",
            points: allPoints.map(p => [p.x, p.y]),
            color: allPoints.map(p => p.color),
            layerId: allPoints.map(p => p.layerId)
        }]
    };

    return fetchWithCSRF("/segment/", {
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
        
        console.log("Segmentation response:", data);

        resetZoom();
        window.lastSegmentationData = data;

        const resultImage = document.getElementById("segmentedResultImage");

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
            const showSegMapBtn = document.getElementById("showSegMapBtn");
            if (showSegMapBtn) {
                showSegMapBtn.style.display = "none";
            }
        }

        const stageContainer = document.getElementById("segmentationStage");
        if (stageContainer) {
            stageContainer.innerHTML = "";
            const width = resultImage.naturalWidth || resultImage.clientWidth;
            const height = resultImage.naturalHeight || resultImage.clientHeight;
            stageContainer.style.width = width + "px";
            stageContainer.style.height = height + "px";
        }

        if (data.segmentation_masks && data.segmentation_masks.individual_masks && data.segmentation_masks.individual_masks.length > 0) {
            createMaskPreviewPanel(data.segmentation_masks);
        }

        if (data.final_mask && data.final_mask.length > 0) {
            console.log('Auto-loading segmentation results as editable annotations');
            loadSegmentationResultsAsAnnotations(data);
            showEditingModeIndicator(true);

            
        }

        // ✅ CHECK FOR PREDICTIONS AND SHOW/HIDE TOGGLE BUTTON
        if (!data.predicted_annotations || data.predicted_annotations.length === 0) {
            console.log("No predicted annotations found");
            hidePredictionToggleButton(); // Hide toggle when no predictions
            redrawAnnotations();
            return;
        }
        
        try {
            state.scribbles = state.scribbles.filter(s => !s.isPrediction);

            const predictedAnnotations = data.predicted_annotations;

            if (Array.isArray(predictedAnnotations)) {
                const processedPoints = [];

                predictedAnnotations.forEach((annotation, index) => {
                    if (annotation && annotation.shape_type === "polygon" && Array.isArray(annotation.points)) {
                        const points = annotation.points.map(p => ({
                            x: p[0] * state.displayScale,
                            y: p[1] * state.displayScale,
                            color: "cyan"
                        }));

                        state.scribbles.push({
                            points: points,
                            isPrediction: true,
                            color: annotation.color ? `rgb(${annotation.color[0]}, ${annotation.color[1]}, ${annotation.color[2]})` : "cyan",
                            class_id: annotation.class_id || "fluid"
                        });
                    }
                    else if (Array.isArray(annotation)) {
                        if (Array.isArray(annotation[0])) {
                            const [[x, y], color] = annotation;           
                            const scaleX = state.originalImageDimensions.width / 512;  
                            const scaleY = state.originalImageDimensions.height / 224; 
                            
                            const scaledX = x * scaleX;
                            const scaledY = y * scaleY;                     
                            processedPoints.push({ 
                                x: scaledX, 
                                y: scaledY, 
                                color: color || "blue"
                            });
                        } else {
                            const [x, y] = annotation;
                            const scaleX = state.originalImageDimensions.width / 512;  
                            const scaleY = state.originalImageDimensions.height / 224; 
                            
                            const scaledX = x * scaleX;
                            const scaledY = y * scaleY;                     
                            processedPoints.push({ 
                                x: scaledX, 
                                y: scaledY, 
                                color: "blue" 
                            });
                        }
                    }
                });

                if (processedPoints.length > 0) {
                    state.scribbles.push({
                        points: processedPoints,
                        isPrediction: true,
                        color: "blue"
                    });
                }

                if (data.class_info) {
                    createClassLegend(data.class_info);
                }
                showPredictionToggleButton();
            }

            redrawAnnotations();
        } catch (err) {
            console.error("Error processing annotations:", err);
            hidePredictionToggleButton(); // Hide toggle on error
            redrawAnnotations();
        }
    })
    .catch(error => {
        console.error("Error in handleAnnotations:", error);
        hidePredictionToggleButton();
        redrawAnnotations();
        alert('Error processing segmentation: ' + error.message);
    });
}

function showPredictionToggleButton() {
    const toggleBtn = document.getElementById('togglePredictionsBtn');
    
    if (toggleBtn) {
        toggleBtn.style.display = 'inline-block';
        const toggleText = document.getElementById('predictionToggleText');
        if (state.showPredictions) {
            toggleBtn.classList.remove('btn-outline-secondary');
            toggleBtn.classList.add('btn-outline-info');
            toggleText.innerHTML = '<i class="fa-solid fa-eye-slash me-2"></i>Hide Contours';
        } else {
            toggleBtn.classList.remove('btn-outline-info');
            toggleBtn.classList.add('btn-outline-secondary');
            toggleText.innerHTML = '<i class="fa-solid fa-eye me-2"></i>Show Contours';
        }
    }
}

function hidePredictionToggleButton() {
    const toggleBtn = document.getElementById('togglePredictionsBtn');
    
    if (toggleBtn) {
        toggleBtn.style.display = 'none';
    }
}