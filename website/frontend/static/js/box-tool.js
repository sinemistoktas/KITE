
import { state } from './state.js';
import { redrawAnnotations } from './canvas-tools.js';
import { createLayer } from './layers.js';
import { screenToImageCoords } from './canvas-utils.js';

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
    
    console.log('Box Tool: Initialized');
}

// Handle box button click
function handleBoxButtonClick() {
    console.log('Box Tool: Button clicked, current mode:', state.mode);
    
    // Toggle box mode following the same pattern as other tools
    if (state.mode === 'box') {
        state.mode = null;
    } else {
        state.mode = 'box';
    }
    
    // Update button styles
    updateBoxButtonStyle();
    
    console.log('Box Tool: Mode set to:', state.mode);
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
    // Only handle if we're actually in box mode
    if (state.mode !== 'box') return false;
    
    console.log('Box Tool: Mouse down');
    
    const coords = screenToImageCoords(e.clientX, e.clientY);
    state.mouseX = coords.x;
    state.mouseY = coords.y;
    
    // Start box drawing
    state.isBoxDrawing = true;
    state.boxStartPoint = { ...coords };
    
    // Create a new layer for the box
    const layerId = createLayer("Box", state.selectedColor);
    state.currentStrokeColor = state.selectedColor;
    state.currentLayerId = layerId;
    
    redrawAnnotations();
    return true;
}

export function handleBoxMouseMove(e) {
    // Only handle if we're actually in box mode and drawing a box
    if (state.mode !== 'box' || !state.isBoxDrawing) return false;
    
    const coords = screenToImageCoords(e.clientX, e.clientY);
    state.mouseX = coords.x;
    state.mouseY = coords.y;
    
    redrawAnnotations();
    return true;
}

export function handleBoxMouseUp() {
    // Only handle if we're actually in box mode and drawing a box
    if (state.mode !== 'box' || !state.isBoxDrawing || !state.boxStartPoint) return false;
    
    console.log('Box Tool: Mouse up - creating box');
    
    // Create rectangle points (clockwise from top-left)
    const boxPoints = [
        { x: state.boxStartPoint.x, y: state.boxStartPoint.y },    // Top-left
        { x: state.mouseX, y: state.boxStartPoint.y },             // Top-right
        { x: state.mouseX, y: state.mouseY },                      // Bottom-right
        { x: state.boxStartPoint.x, y: state.mouseY },             // Bottom-left
        { x: state.boxStartPoint.x, y: state.boxStartPoint.y }     // Close the shape
    ];
    
    // Make sure we're not creating a box that's too small
    const width = Math.abs(state.mouseX - state.boxStartPoint.x);
    const height = Math.abs(state.mouseY - state.boxStartPoint.y);
    
    if (width < 5 || height < 5) {
        console.log('Box Tool: Box too small, canceling');
        state.isBoxDrawing = false;
        state.boxStartPoint = null;
        return true;
    }
    
    // Add to scribbles - this should make it persist
    const newBox = {
        points: boxPoints,
        isPrediction: false,
        color: state.currentStrokeColor,
        layerId: state.currentLayerId,
        isBox: true // Flag to identify box annotations
    };
    
    state.scribbles.push(newBox);
    
    console.log('Box Tool: Box added to scribbles array:', 
                state.scribbles.length, 
                'Current box:', newBox);
    
    // Reset drawing state
    state.isBoxDrawing = false;
    state.boxStartPoint = null;
    
    redrawAnnotations();
    return true;
}

export function handleBoxMouseLeave() {
    // Only handle if we're actually in box mode and drawing a box
    if (state.mode !== 'box' || !state.isBoxDrawing) return false;
    
    console.log('Box Tool: Mouse leave - canceling box');
    
    // Cancel box drawing
    state.isBoxDrawing = false;
    state.boxStartPoint = null;
    
    redrawAnnotations();
    return true;
}

// Draw box preview during drawing
export function drawBoxPreview(ctx, adjustForZoom, restoreZoom) {
    if (!state.isBoxDrawing || !state.boxStartPoint) return;
    
    adjustForZoom(ctx);
    
    ctx.strokeStyle = state.selectedColor;
    ctx.lineWidth = 2 / state.zoomLevel;
    
    // Calculate box dimensions
    const width = state.mouseX - state.boxStartPoint.x;
    const height = state.mouseY - state.boxStartPoint.y;
    
    // Draw rectangle
    ctx.beginPath();
    ctx.rect(state.boxStartPoint.x, state.boxStartPoint.y, width, height);
    ctx.stroke();
    
    restoreZoom(ctx);
}