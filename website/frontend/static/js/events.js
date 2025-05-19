import { state } from './state.js';
import { screenToImageCoords } from './canvas-utils.js';
import { redrawAnnotations } from './canvas-tools.js';
import { createLayer } from './layers.js';
import { toggleFillTool, handleFillToolClick, resetFillTool, updateFillToolStatus } from './fill-tool.js';
import { zoomToPoint, zoomOut, resetZoom } from './canvas-tools.js';
import { handleBoxMouseDown, handleBoxMouseMove, handleBoxMouseUp, handleBoxMouseLeave } from './box-tool.js';


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
    // Add box mode button handler - let the box-tool.js handle this directly
    // document.getElementById('boxMode')?.addEventListener('click', () => setMode('box'));

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

    // Mouse events
    annotationCanvas.addEventListener('mousedown', (e) => {
        const coords = screenToImageCoords(e.clientX, e.clientY);
        state.mouseX = coords.x;
        state.mouseY = coords.y;

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
        // IMPORTANT: Only let the box tool handle it if we're in box mode
        if (state.mode === 'box' && handleBoxMouseUp()) {
            return; // Box tool handled it, stop processing
        }

        // line
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

        // fill tool
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
        // IMPORTANT: Only let the box tool handle it if we're in box mode
        if (state.mode === 'box' && handleBoxMouseLeave()) {
            return; // Box tool handled it, stop processing
        }

        // line
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
        // Note: Box keyboard shortcut "B" is handled in box-tool.js
    });
}

function setMode(newMode) {
    const zoomBtn = document.getElementById('zoomInBtn');

    // Exit zoom mode if switching to a drawing mode
    if (state.zoomMode) {
        state.zoomMode = false;
        zoomBtn?.classList.remove('btn-primary');
        zoomBtn?.classList.add('btn-outline-primary');
        state.annotationCanvas.classList.remove("zoom-cursor");
        state.annotationCanvas.style.cursor = "crosshair"; // reset
    }

    if (state.isFillToolActive) {
        state.isFillToolActive = false;
        document.getElementById('fillToolBtn')?.classList.remove('btn-danger');
        document.getElementById('fillToolBtn')?.classList.add('btn-outline-danger');
        document.getElementById('fillToolStatus')?.style.setProperty('display', 'none');
        resetFillTool();
    }

    if (newMode === "eraseAll") {
        eraseAllAnnotations();
        updateButtonStyles(); 
        return; 
    }

    if (state.showEraserCursor && state.mode === "eraser") {
        state.annotationCanvas.style.cursor = "none"; // ✅ invisible cursor
    } else if (!state.zoomMode && !state.isFillToolActive) {
        state.annotationCanvas.style.cursor = "crosshair";
    }

    if (newMode === "fill") {
        if (state.zoomMode) {
            state.zoomMode = false;
            document.getElementById("zoomInBtn")?.classList.remove("btn-primary");
            document.getElementById("zoomInBtn")?.classList.add("btn-outline-primary");
            state.annotationCanvas.classList.remove("zoom-cursor");
        }
    
        state.isFillToolActive = !state.isFillToolActive;
    
        if (state.isFillToolActive) {
            resetFillTool();
            updateFillToolStatus("Draw a closed boundary around the area you want to fill");
        } else {
            document.getElementById("fillToolStatus")?.style.setProperty("display", "none");
        }
    
        state.mode = state.isFillToolActive ? "fill" : null;
        state.annotationCanvas.style.cursor = state.isFillToolActive ? "crosshair" : "crosshair";
    
        updateButtonStyles();
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
    
    if (state.mode === 'eraser' && state.showEraserCursor) {
        state.annotationCanvas.style.cursor = "none";
    } else if (!state.zoomMode && !state.isFillToolActive) {
        state.annotationCanvas.style.cursor = "crosshair";
    }
    
    updateButtonStyles();
    redrawAnnotations();
}

function toggleZoomMode() {
    state.zoomMode = !state.zoomMode;
    const zoomBtn = document.getElementById('zoomInBtn');

    if (state.zoomMode) {
        state.mode = null;
        state.showEraserCursor = false;
        state.annotationCanvas.style.cursor = "zoom-in";
        zoomBtn?.classList.add("btn-primary");
        zoomBtn?.classList.remove("btn-outline-primary");
    } else {
        state.annotationCanvas.style.cursor = "crosshair";
        zoomBtn?.classList.remove("btn-primary");
        zoomBtn?.classList.add("btn-outline-primary");
    }

    redrawAnnotations();
}

function eraseAt(x, y) {
    const newScribbles = [];
    const existingLayerIds = new Set(); // Track all layers
    const remainingLayerIds = new Set(); // Track layers that survive

    // Track existing layers first
    for (const stroke of state.scribbles) {
        if (stroke.layerId) {
            existingLayerIds.add(stroke.layerId);
        }
    }

    for (const stroke of state.scribbles) {
        if (stroke.isPrediction) {
            newScribbles.push(stroke);
            continue;
        }

        const hasPointInEraser = stroke.points.some(p => Math.hypot(p.x - x, p.y - y) < 10);
        if (!hasPointInEraser) {
            newScribbles.push(stroke);
            if (stroke.layerId) remainingLayerIds.add(stroke.layerId);
            continue;
        }

        // Try splitting stroke into partials
        let segment = [];
        for (const point of stroke.points) {
            const within = Math.hypot(point.x - x, point.y - y) < 10;
            if (!within) {
                segment.push(point);
            } else {
                if (segment.length > 1) {
                    newScribbles.push({
                        points: segment,
                        isPrediction: false,
                        color: stroke.color,
                        layerId: stroke.layerId
                    });
                    if (stroke.layerId) remainingLayerIds.add(stroke.layerId);
                }
                segment = [];
            }
        }

        if (segment.length > 1) {
            newScribbles.push({
                points: segment,
                isPrediction: false,
                color: stroke.color,
                layerId: stroke.layerId
            });
            if (stroke.layerId) remainingLayerIds.add(stroke.layerId);
        }
    }

    // Remove layers that no longer have any strokes
    for (const layerId of existingLayerIds) {
        if (!remainingLayerIds.has(layerId)) {
            const el = document.getElementById(layerId);
            if (el) el.remove();
            state.visibleLayerIds = state.visibleLayerIds.filter(id => id !== layerId);
        }
    }

    state.scribbles = newScribbles;
    redrawAnnotations();
}

function eraseAllAnnotations() {
    state.scribbles = state.scribbles.filter(s => s.isPrediction); // Keep predictions
    state.layerCounter = 0;
    state.visibleLayerIds = [];
    state.currentLayerId = null;

    const layersContainer = document.getElementById('layersContainer');
    while (layersContainer.firstChild) {
        layersContainer.removeChild(layersContainer.firstChild);
    }

    redrawAnnotations();
}

function updateButtonStyles() {
    const buttons = {
        line: document.getElementById("scribbleMode"),
        dot: document.getElementById("dotMode"),
        eraser: document.getElementById("eraserMode"),
        eraseAll: document.getElementById("eraseAllMode"),
        fill: document.getElementById("fillToolBtn"),
        zoom: document.getElementById("zoomInBtn")
    };
    
    for (const [tool, btn] of Object.entries(buttons)) {
        if (!btn) continue;
        const isActive =
            (tool === state.mode) ||
            (tool === "fill" && state.isFillToolActive) ||
            (tool === "zoom" && state.zoomMode);
    
        btn.classList.toggle("btn-danger", isActive && tool !== "zoom");
        btn.classList.toggle("btn-outline-danger", !isActive && tool !== "zoom");
    
        if (tool === "zoom") {
            btn.classList.toggle("btn-primary", isActive);
            btn.classList.toggle("btn-outline-primary", !isActive);
        }
    }
}