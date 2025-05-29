import { state } from './state.js';
import { redrawAnnotations } from './canvas-tools.js';
import { createLayer } from './layers.js';
import { screenToImageCoords } from './canvas-utils.js';

// Initialize box tool
export function initBoxTool() {
    console.log('Box Tool: Initializing');
    
    
    console.log('Box Tool: Initialized');
}

export function handleBoxMouseDown(e) {
    if (state.mode !== 'box') return false;
    
    console.log('Box Tool: Mouse down');
    
    const coords = screenToImageCoords(e.clientX, e.clientY);
    state.mouseX = coords.x;
    state.mouseY = coords.y;
    
    state.isBoxDrawing = true;
    state.boxStartPoint = { ...coords };
    
    const layerId = createLayer("Box", state.selectedColor);
    state.currentStrokeColor = state.selectedColor;
    state.currentLayerId = layerId;
    
    redrawAnnotations();
    return true;
}

export function handleBoxMouseMove(e) {
    if (state.mode !== 'box' || !state.isBoxDrawing) return false;
    
    const coords = screenToImageCoords(e.clientX, e.clientY);
    state.mouseX = coords.x;
    state.mouseY = coords.y;
    
    redrawAnnotations();
    return true;
}

export function handleBoxMouseUp() {
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