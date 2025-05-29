import { state } from './state.js';
import { screenToImageCoords } from './canvas-utils.js';
import { redrawAnnotations } from './canvas-tools.js';
import { createLayer } from './layers.js';
import { toggleFillTool, handleFillToolClick, resetFillTool, updateFillToolStatus } from './fill-tool.js';
import { zoomToPoint, zoomOut, resetZoom } from './canvas-tools.js';
import { handleBoxMouseDown, handleBoxMouseMove, handleBoxMouseUp, handleBoxMouseLeave } from './box-tool.js';
import { initializeUNetPredictions } from './api-service.js';

export function bindUIEvents() {
    const { annotationCanvas } = state;

    // Buttons
    document.getElementById('scribbleMode')?.addEventListener('click', () => setMode('line'));
    document.getElementById('dotMode')?.addEventListener('click', () => setMode('dot'));
    document.getElementById('eraserMode')?.addEventListener('click', () => setMode('eraser'));
    document.getElementById('eraseAllMode')?.addEventListener('click', () => setMode('eraseAll'));
    document.getElementById('fillToolBtn')?.addEventListener('click', () => {
        setMode(state.mode === "fill" ? null : "fill");
    });
    
    // Box mode button - let the box-tool.js handle this directly
    document.getElementById('boxMode')?.addEventListener('click', () => setMode('box'));
    document.addEventListener('modeChanged', (e) => {
        updateButtonStyles();
    });

    document.getElementById('zoomInBtn')?.addEventListener('click', () => {
        if (state.zoomMode) {
            const coords = screenToImageCoords(state.mouseX, state.mouseY);
            zoomToPoint(coords.x, coords.y);
        } else {
            toggleZoomMode();
        }
    });

    document.getElementById('zoomOutBtn')?.addEventListener('click', zoomOut);
    document.getElementById('resetZoomBtn')?.addEventListener('click', resetZoom);

    // Method selector (for legacy segmentation method system)
    const traditionalMethod = document.getElementById('traditionalMethod');
    const unetMethod = document.getElementById('unetMethod');

    if (traditionalMethod && unetMethod) {
        traditionalMethod.addEventListener('change', () => {
            state.unetMode = false;
            updateMethodDescription();
        });

        unetMethod.addEventListener('change', () => {
            state.unetMode = true;
            updateMethodDescription();

            if (state.imageName) {
                initializeUNetPredictions(state.imageName)
                    .then(predictions => {
                        if (predictions.length > 0) {
                            // Use the full predictions data with class info
                            initializeAnnotationsFromPredictions(predictions);
                        }
                    });
            }
        });

        if (state.segmentationMethod === "unet") {
            state.unetMode = true;
            unetMethod.checked = true;
        } else {
            state.unetMode = false;
            traditionalMethod.checked = true;
        }

        updateMethodDescription();
    }

    // Mouse events
    annotationCanvas.addEventListener('mousedown', (e) => {
        const coords = screenToImageCoords(e.clientX, e.clientY);
        state.mouseX = coords.x;
        state.mouseY = coords.y;

        // Handle box tool events first (for MedSAM bounding boxes)
        if (handleBoxMouseDown(e)) return;

        if (state.zoomMode) {
            zoomToPoint(coords.x, coords.y);
            return;
        }

        if (state.isFillToolActive) {
            handleFillToolClick(coords);
            return;
        }

        if (state.mode === 'dot') {
            const layerId = createLayer("Dot", state.selectedColor);
            state.scribbles.push({
                points: [coords],
                isPrediction: false,
                color: state.selectedColor,
                layerId
            });
            redrawAnnotations();
        } else if (state.mode === 'line') {
            state.isDrawing = true;
            const layerId = createLayer("Line", state.selectedColor);
            state.currentStroke = [coords];
            state.currentStrokeColor = state.selectedColor;
            state.currentLayerId = layerId;
        } else if (state.mode === 'eraser') {
            state.isErasing = true;
            eraseAt(coords.x, coords.y);
        }
    });

    annotationCanvas.addEventListener('mousemove', (e) => {
        // Handle box tool events first
        if (handleBoxMouseMove(e)) return;

        const coords = screenToImageCoords(e.clientX, e.clientY);
        state.mouseX = coords.x;
        state.mouseY = coords.y;

        if (state.isFillToolActive && state.isDrawingBoundary) {
            state.currentBoundary.push(coords);
            redrawAnnotations();
        }

        if (state.isDrawing && state.mode === 'line') {
            state.currentStroke.push(coords);
            redrawAnnotations();
        }

        if (state.mode === 'eraser') {
            if (state.isErasing) {
                eraseAt(coords.x, coords.y);
            } else {
                redrawAnnotations();
            }
        }
    });

    annotationCanvas.addEventListener('mouseup', () => {
        // First check if box tool handles this event
        if (state.mode === 'box' && handleBoxMouseUp()) {
            return; // Box tool handled it, stop processing
        }

        // Handle line drawing
        if (state.isDrawing && state.currentStroke.length > 0) {
            state.scribbles.push({
                points: state.currentStroke,
                isPrediction: false,
                color: state.currentStrokeColor,
                layerId: state.currentLayerId
            });
            state.currentStroke = [];
            redrawAnnotations();
        }

        // Handle fill tool
        if (state.isFillToolActive && state.isDrawingBoundary) {
            state.isDrawingBoundary = false;
            if (state.currentBoundary.length >= 3) {
                state.currentBoundary.push({ ...state.currentBoundary[0] });
                state.boundaryComplete = true;
                updateFillToolStatus("Now click inside the boundary to fill it");
            } else {
                state.currentBoundary = [];
                updateFillToolStatus("Please draw a closed boundary with at least 3 points");
            }
            redrawAnnotations();
        }

        state.isDrawing = false;
        state.isErasing = false;
    });

    annotationCanvas.addEventListener('mouseleave', () => {
        // First check if box tool handles this event
        if (state.mode === 'box' && handleBoxMouseLeave()) {
            return;
        }

        // Handle line drawing
        if (state.isDrawing && state.currentStroke.length > 0) {
            state.scribbles.push({
                points: state.currentStroke,
                isPrediction: false,
                color: state.currentStrokeColor,
                layerId: state.currentLayerId
            });
            state.currentStroke = [];
            redrawAnnotations();
        }

        state.isDrawing = false;
        state.isErasing = false;
    });

    // Keyboard Shortcuts
    document.addEventListener('keydown', (e) => {
        if (["INPUT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
        const key = e.key.toUpperCase();

        if (key === "L") setMode(state.mode === "line" ? null : "line");
        else if (key === "E") setMode(state.mode === "eraser" ? null : "eraser");
        else if (key === "D") setMode(state.mode === "dot" ? null : "dot");
        else if (key === "A") setMode(state.mode === "eraseAll" ? null : "eraseAll");
        else if (key === "Z") toggleZoomMode();
        else if (key === "F") toggleFillTool();
        else if (key === "B") setMode(state.mode === "box" ? null : "box");
    });

    // Segmentation mode form handling (for legacy system)
    const uploadForm = document.querySelector('form[enctype="multipart/form-data"]');

    if (uploadForm) {
        uploadForm.addEventListener('submit', function (event) {
            const formData = new FormData(this);
            const method = formData.get('segmentation_method');
            if (method) {
                state.segmentationMethod = method;
                state.unetMode = method === 'unet';
            }
        });
    }
}

export function updateMethodDescription() {
    const descriptionElement = document.getElementById('methodDescription');
    if (!descriptionElement) return;

    const unetMethod = document.getElementById('unetMethod');

    if (unetMethod && unetMethod.checked) {
        descriptionElement.innerHTML =
            '<div class="alert alert-info">' +
            '<i class="fa-solid fa-robot me-2"></i> ' +
            '<strong>UNet Assisted Mode:</strong> Initial segmentation is generated automatically by the UNet model. ' +
            'You can then refine the results using the annotation tools.' +
            '</div>';
    } else {
        descriptionElement.innerHTML =
            '<div class="alert alert-info">' +
            '<i class="fa-solid fa-pencil me-2"></i> ' +
            '<strong>Traditional Mode:</strong> Semi-automated segmentation using manual annotations. ' +
            'Draw annotations to indicate regions of interest.' +
            '</div>';
    }
}

function setMode(newMode) {
    const zoomBtn = document.getElementById('zoomInBtn');

    // Exit zoom mode if switching to a drawing mode
    if (state.zoomMode && newMode !== null) {
        state.zoomMode = false;
        zoomBtn?.classList.remove('btn-primary');
        zoomBtn?.classList.add('btn-outline-primary');
        state.annotationCanvas.classList.remove("zoom-cursor");
    }

    if (state.isFillToolActive && newMode !== "fill") {
        state.isFillToolActive = false;
        document.getElementById('fillToolBtn')?.classList.remove('btn-danger');
        document.getElementById('fillToolBtn')?.classList.add('btn-outline-danger');
        document.getElementById('fillToolStatus')?.style.setProperty('display', 'none');
        resetFillTool();
    }

    if (newMode === "eraseAll") {
        eraseAllAnnotations();
        state.mode = null; // Clear mode after erasing all.
        updateButtonStyles();
        setCursor();
        return; 
    }

    if (newMode === "fill") {
        if (state.zoomMode) {
            state.zoomMode = false;
            document.getElementById("zoomInBtn")?.classList.remove("btn-primary");
            document.getElementById("zoomInBtn")?.classList.add("btn-outline-primary");
            state.annotationCanvas.classList.remove("zoom-cursor");
        }

        state.isFillToolActive = !state.isFillToolActive;
    
        const fillBtn = document.getElementById('fillToolBtn');
        if (state.isFillToolActive) {
            fillBtn?.classList.add('btn-danger');
            fillBtn?.classList.remove('btn-outline-danger');
            resetFillTool();
            updateFillToolStatus("Draw a closed boundary around the area you want to fill");
        } else {
            fillBtn?.classList.remove('btn-danger');
            fillBtn?.classList.add('btn-outline-danger');
            document.getElementById('fillToolStatus')?.style.setProperty('display', 'none');
        }
    
        state.mode = state.isFillToolActive ? "fill" : null;
        
        updateButtonStyles();
        setCursor();
        redrawAnnotations();
        return;
    }

    if (state.mode === newMode) {
        state.mode = null;
        state.showEraserCursor = false;
    } else {
        state.mode = newMode;
        state.showEraserCursor = (newMode === 'eraser');
    }
    

    updateButtonStyles();
    setCursor();
    redrawAnnotations();
}

function toggleZoomMode() {
    state.zoomMode = !state.zoomMode;
    const zoomBtn = document.getElementById('zoomInBtn');

    if (state.zoomMode) {
        state.mode = null;
        state.showEraserCursor = false;
        state.isFillToolActive = false;

        document.getElementById('fillToolBtn')?.classList.remove('btn-danger');
        document.getElementById('fillToolBtn')?.classList.add('btn-outline-danger');
        document.getElementById('fillToolStatus')?.style.setProperty('display', 'none');
        resetFillTool();
        
        state.annotationCanvas.style.cursor = "zoom-in";
        zoomBtn?.classList.add("btn-primary");
        zoomBtn?.classList.remove("btn-outline-primary");
    } else {
        zoomBtn?.classList.remove("btn-primary");
        zoomBtn?.classList.add("btn-outline-primary");
    }

    updateButtonStyles();
    setCursor();
    redrawAnnotations();
}
// Helper function to set the cursor.
function setCursor() {
    if (state.zoomMode) {
        state.annotationCanvas.style.cursor = "zoom-in";
    } else if (state.mode === 'eraser' && state.showEraserCursor) {
        state.annotationCanvas.style.cursor = "none";
    } else {
        state.annotationCanvas.style.cursor = "crosshair";
    }
}

function updateButtonStyles() {
    const buttons = {
        line: document.getElementById("scribbleMode"),
        dot: document.getElementById("dotMode"),
        eraser: document.getElementById("eraserMode"),
        eraseAll: document.getElementById("eraseAllMode"),
        fill: document.getElementById("fillToolBtn"),
        zoom: document.getElementById("zoomInBtn"),
        box: document.getElementById("boxMode")
    };
    
    for (const [tool, btn] of Object.entries(buttons)) {
        if (!btn) continue;
        
        const isActive =
            (tool === state.mode) ||
            (tool === "fill" && state.isFillToolActive) ||
            (tool === "zoom" && state.zoomMode);
        //TODO: the zoom in button should be selected.
        if (tool === "zoom") {
            btn.classList.toggle("btn-primary", isActive);
            btn.classList.toggle("btn-outline-primary", !isActive);
        } 
        else {
            btn.classList.toggle("btn-danger", isActive);
            btn.classList.toggle("btn-outline-danger", !isActive);
        }
    }
}

window.updateButtonStyles = updateButtonStyles;

function eraseAt(x, y) {
    const newScribbles = [];
    const existingLayerIds = new Set(); // Track all layers
    const remainingLayerIds = new Set(); // Track layers that survive
    const eraseRadius = 10;

    for (const stroke of state.scribbles) {
        if (stroke.layerId) {
            existingLayerIds.add(stroke.layerId);
        }
    }
    let erasedCount = 0;
    let protectedCount = 0;
    let strokesProcessed = 0;

    for (const stroke of state.scribbles) {
        strokesProcessed++;
        // Always keep predictions contours - they shouldn't be erasable by the user!
        if (stroke.isPrediction) {
            newScribbles.push(stroke);
            continue;
        }

        // IF the stroke belongs to a layer that is currently NOT visible, it should NOT be erased. This seems better.
        const isLayerVisible = !stroke.layerId || state.visibleLayerIds.includes(stroke.layerId);
        
        if (!isLayerVisible) {
            newScribbles.push(stroke);
            if (stroke.layerId) remainingLayerIds.add(stroke.layerId);
    
            const wouldBeErased = stroke.points.some(p => Math.hypot(p.x - x, p.y - y) < eraseRadius);
            if (wouldBeErased) {
                protectedCount++;
            }
            continue;
        }
        const pointsInEraser = stroke.points.filter(p => Math.hypot(p.x - x, p.y - y) < eraseRadius);
        const hasPointInEraser = pointsInEraser.length > 0;
        if (!hasPointInEraser) {
            newScribbles.push(stroke);
            if (stroke.layerId) remainingLayerIds.add(stroke.layerId);
            continue;
        }

        if (stroke.isLoadedAnnotation || stroke.isDot || stroke.isSegmentationResult || stroke.points.length === 1) {
            const point = stroke.points[0];
            const withinEraser = Math.hypot(point.x - x, point.y - y) < eraseRadius;
            if (!withinEraser) {
                newScribbles.push(stroke);
                if (stroke.layerId) remainingLayerIds.add(stroke.layerId);
            } else {
                erasedCount++;
            }
        } 
        else if (stroke.isBox) {
            erasedCount++;
        }
        else if (stroke.isFilled || stroke.type === 'fill') {
            erasedCount++;
        }
        else {
            let segments = [];
            let currentSegment = [];
            let pointsErased = 0;
            
            for (const point of stroke.points) {
                const withinEraser = Math.hypot(point.x - x, point.y - y) < eraseRadius;
                
                if (!withinEraser) {
                    currentSegment.push(point);
                } else {
                    pointsErased++;
                    if (currentSegment.length > 1) {
                        segments.push([...currentSegment]);
                    }
                    currentSegment = [];
                }
            }
            
            if (currentSegment.length > 1) {
                segments.push(currentSegment);
            }
            for (const segment of segments) {
                newScribbles.push({
                    points: segment,
                    isPrediction: false,
                    color: stroke.color,
                    layerId: stroke.layerId,
                    isLoadedAnnotation: stroke.isLoadedAnnotation || false,
                    isDot: false,
                    isSegmentationResult: stroke.isSegmentationResult || false,
                    isBox: false,
                    isFilled: stroke.isFilled || false,
                    type: stroke.type
                });
                if (stroke.layerId) remainingLayerIds.add(stroke.layerId);
            }
            
            if (pointsErased > 0) {
                erasedCount += pointsErased;
            }
        }
    }

    // Remove layers that no longer have any strokes, BUT don't remove hidden layers!!!
    const layersRemoved = [];
    for (const layerId of existingLayerIds) {
        if (!remainingLayerIds.has(layerId)) {
            const wasVisible = state.visibleLayerIds.includes(layerId);
            if (wasVisible) {
                const layerElement = document.getElementById(layerId);
                if (layerElement) {
                    layerElement.remove();
                    layersRemoved.push(layerId);
                }
                state.visibleLayerIds = state.visibleLayerIds.filter(id => id !== layerId);
            }
        }
    }

    state.scribbles = newScribbles;
    redrawAnnotations();
}

function eraseAllAnnotations() {
    state.scribbles = state.scribbles.filter(s => s.isPrediction);
    state.layerCounter = 0;
    state.visibleLayerIds = [];
    state.currentLayerId = null;

    const layersContainer = document.getElementById('layersContainer');
    if (layersContainer) {
        while (layersContainer.firstChild) {
            layersContainer.removeChild(layersContainer.firstChild);
        }
    }

    redrawAnnotations();
}

export function initializeAnnotationsFromPredictions(predictions) {
    if (!predictions || !predictions.length) return;

    import('./layers.js').then(module => {
        const { createLayer } = module;

        const predictionsByClass = {};
        
        predictions.forEach(shape => {
            if (shape.points && shape.points.length > 0) {
                const classId = shape.class_id || 1;
                if (!predictionsByClass[classId]) {
                    predictionsByClass[classId] = [];
                }
                predictionsByClass[classId].push(shape);
            }
        });
        
        Object.entries(predictionsByClass).forEach(([classId, shapes]) => {
            const layerName = `Layer ${classId}`;
            const color = shapes[0].color ? 
                `rgb(${shapes[0].color[0]}, ${shapes[0].color[1]}, ${shapes[0].color[2]})` : 
                "#ff0000";
                
            const layerId = createLayer(layerName, color);
            
            shapes.forEach(shape => {
                const points = shape.points.map(point => ({
                    x: point[0],
                    y: point[1]
                }));
                
                state.scribbles.push({
                    points: points,
                    isPrediction: false, // Treat as a normal annotation that can be edited
                    color: color,
                    layerId: layerId,
                    class_id: parseInt(classId)
                });
            });
        });

        import('./canvas-tools.js').then(module => {
            const { redrawAnnotations } = module;
            redrawAnnotations();
        });
    });
}

export function processUNetPredictions(predictedPoints) {
    if (!predictedPoints || predictedPoints.length === 0) return [];

    return predictedPoints.flatMap(shape => {
        if (shape.shape_type === "polygon" && shape.points && shape.points.length > 0) {
            // For polygon shapes with points array
            return shape.points.map(point => ({
                x: point[0],
                y: point[1],
                color: shape.color ? shape.color[0] : "#ff0000"
            }));
        } else if (Array.isArray(shape)) {
            // For simple point arrays
            if (Array.isArray(shape[0])) {
                const [coords, color] = shape;
                return {
                    x: coords[0], 
                    y: coords[1],
                    color: color || "#ff0000"
                };
            } else {
                return {
                    x: shape[0],
                    y: shape[1],
                    color: "#ff0000"
                };
            }
        }
        return null;
    }).filter(Boolean);
}