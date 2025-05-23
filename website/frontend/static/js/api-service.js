import { state } from './state.js';
import { redrawAnnotations, resetZoom } from './canvas-tools.js';

export function getCSRFToken(name = "csrftoken") {
    const value = `; ${document.cookie}`;
    const parts = value.split(`; ${name}=`);
    return parts.length === 2 ? parts.pop().split(';').shift() : '';
}

export function handleAnnotations() {
    const existingPanel = document.getElementById('maskPreviewPanel');
    if (existingPanel) {
        existingPanel.remove();
    }
    
    state.scribbles = state.scribbles.filter(s => !s.isPrediction);
    redrawAnnotations();

    const allPoints = state.scribbles.flatMap(s =>
        s.points.map(p => ({
            x: p.x,
            y: p.y,
            color: s.color,
            layerId: s.layerId  // Include layer ID.
        }))
    );

    const payload = {
        image_name: state.imageName,
        shapes: [{
            label: "anomaly",
            points: allPoints.map(p => [p.x, p.y]),
            color: allPoints.map(p => p.color),
            layerId: allPoints.map(p => p.layerId)  // Send layer IDs.
        }]
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
        resetZoom();

        // Store segmentation data globally for bulk download.
        window.lastSegmentationData = data;

        // Added for downloading the general segmentation mask 
        console.log(data.segmentation_mask_npy);
        console.log("Segment response:", data);

        const resultImage = document.getElementById("segmentedResultImage");
        resultImage.src = `data:image/png;base64,${data.segmented_image}`;

        document.getElementById("segmentationResult").style.display = "block";
        const downloadBtn = document.getElementById("downloadSegmentedImage");
        downloadBtn.href = resultImage.src;
        downloadBtn.download = "segmented_result.png";
        downloadBtn.style.display = "inline-block";

        const stageContainer = document.getElementById("segmentationStage");
        stageContainer.innerHTML = ""; // Clear previous content

        const width = resultImage.naturalWidth || resultImage.clientWidth;
        const height = resultImage.naturalHeight || resultImage.clientHeight;
        stageContainer.style.width = width + "px";
        stageContainer.style.height = height + "px";

        if (data.segmentation_masks && data.segmentation_masks.individual_masks && data.segmentation_masks.individual_masks.length > 0) {
            createMaskPreviewPanel(data.segmentation_masks);
        }

        const stage = new Konva.Stage({
            container: "segmentationStage",
            width: width,
            height: height,
        });

        const layer = new Konva.Layer();
        stage.add(layer);

        // Current problems noted so we wouldn't forget
        // 1. the deleted regions are not removed from the canvas since it takes from the backend, it needs to be refreshed every time 
        // 2. it works a bit slow when there are too many regions on the image, try other alternatives rather than grouping them all
        // 3. canvas tools and events should be updated accordingly to support this function 
        (data.final_mask || []).forEach(({ regionId, pixels, color }) => {
            const group = new Konva.Group({
                id: regionId,
                draggable: true,
            });

            pixels.forEach(([y, x]) => {
                const rect = new Konva.Rect({
                    x: x,
                    y: y,
                    width: 1,
                    height: 1,
                    fill: color || "rgba(233, 37, 37, 0.98)",
                });
                group.add(rect);
            });

            group.on("click", () => {
                console.log(`Clicked region: ${regionId}`);
            });

            layer.add(group);
        });

        layer.draw();

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

// Function to create the "segmentation masks" preview panel.
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
    flexContainer.appendChild(maskPanel);
    
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
            
            const response = await fetch("/bulk-download-masks/", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "X-CSRFToken": getCSRFToken()
                },
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