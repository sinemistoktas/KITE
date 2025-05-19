// box-tool.js
import { state } from './state.js';
import { redrawAnnotations } from './canvas-tools.js';
import { createLayer } from './layers.js';
import { screenToImageCoords } from './canvas-utils.js';

// Box state variables
let isBoxDrawing = false;
let boxStartPoint = null;

// Initialize box tool
export function initBoxTool() {
    console.log('Box Tool: Initializing');
    
    // Add button click handler
    const boxButton = document.getElementById('boxMode');
    if (boxButton) {
        boxButton.addEventListener('click', handleBoxButtonClick);
    }
    
    // Add keyboard shortcut
    document.addEventListener('keydown', (e) => {
        if (["INPUT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
        if (e.key.toUpperCase() === "B") {
            handleBoxButtonClick();
        }
    });
}

// Handle box button click
function handleBoxButtonClick() {
    // Toggle box mode following the same pattern as other tools
    if (state.mode === 'box') {
        state.mode = null;
    } else {
        state.mode = 'box';
    }
    
    // Update button styles
    updateBoxButtonStyle();
}

// Update box button style based on mode
function updateBoxButtonStyle() {
    const boxButton = document.getElementById('boxMode');
    if (!boxButton) return;
    
    if (state.mode === 'box') {
        boxButton.classList.remove('btn-outline-danger');
        boxButton.classList.add('btn-danger');
    } else {
        boxButton.classList.remove('btn-danger');
        boxButton.classList.add('btn-outline-danger');
    }
}

// Mouse event handlers
export function handleBoxMouseDown(e) {
    if (state.mode !== 'box') return false;
    
    const coords = screenToImageCoords(e.clientX, e.clientY);
    state.mouseX = coords.x;
    state.mouseY = coords.y;
    
    // Start box drawing
    isBoxDrawing = true;
    boxStartPoint = { ...coords };
    
    // Create a new layer for the box
    const layerId = createLayer("Box", state.selectedColor);
    state.currentStrokeColor = state.selectedColor;
    state.currentLayerId = layerId;
    
    redrawAnnotations();
    return true;
}

export function handleBoxMouseMove(e) {
    if (state.mode !== 'box' || !isBoxDrawing) return false;
    
    const coords = screenToImageCoords(e.clientX, e.clientY);
    state.mouseX = coords.x;
    state.mouseY = coords.y;
    
    redrawAnnotations();
    return true;
}

export function handleBoxMouseUp(e) {
    if (state.mode !== 'box' || !isBoxDrawing || !boxStartPoint) return false;
    
    const coords = screenToImageCoords(e.clientX, e.clientY);
    
    // Create rectangle points (clockwise from top-left)
    const boxPoints = [
        { x: boxStartPoint.x, y: boxStartPoint.y },    // Top-left
        { x: coords.x, y: boxStartPoint.y },           // Top-right
        { x: coords.x, y: coords.y },                  // Bottom-right
        { x: boxStartPoint.x, y: coords.y },           // Bottom-left
        { x: boxStartPoint.x, y: boxStartPoint.y }     // Close the shape
    ];
    
    // Add to scribbles
    state.scribbles.push({
        points: boxPoints,
        isPrediction: false,
        color: state.currentStrokeColor,
        layerId: state.currentLayerId,
        isBox: true // Flag to identify box annotations
    });
    
    // Reset drawing state
    isBoxDrawing = false;
    boxStartPoint = null;
    
    redrawAnnotations();
    return true;
}

export function handleBoxMouseLeave() {
    if (state.mode !== 'box' || !isBoxDrawing) return false;
    
    // Cancel box drawing
    isBoxDrawing = false;
    boxStartPoint = null;
    
    redrawAnnotations();
    return true;
}

// Function to draw box preview during drawing
export function drawBoxPreview(ctx, adjustForZoom, restoreZoom) {
    if (!isBoxDrawing || !boxStartPoint) return;
    
    adjustForZoom(ctx);
    
    ctx.strokeStyle = state.selectedColor;
    ctx.lineWidth = 2 / state.zoomLevel;
    
    // Calculate box dimensions
    const width = state.mouseX - boxStartPoint.x;
    const height = state.mouseY - boxStartPoint.y;
    
    // Draw rectangle
    ctx.beginPath();
    ctx.rect(boxStartPoint.x, boxStartPoint.y, width, height);
    ctx.stroke();
    
    restoreZoom(ctx);
}