// website/frontend/static/js/medsam-tool.js

/**
 * Simplified MedSAM Tool Integration for KITE
 * Fixed version that works with the current setup
 */

// Global variables
let medsamCanvas;
let medsamCtx;
let uploadedImage;
let isDrawingMedsamBox = false;
let startX = 0;
let startY = 0;
let imageNaturalSize = { width: 0, height: 0 };
let canvasSize = { width: 0, height: 0 };
let currentBoxes = []; // Store all drawn boxes
let segmentationResults = []; // Store all segmentation results
let cumulativeOverlayImage = null; // Store the cumulative overlay
let pendingSegmentation = false; // Flag to track if we need to update segmentation
let undoHistory = []; // Store history of undone boxes and their segmentations

// Color palette for boxes and segmentations
const colorPalette = [
  { box: 'rgba(251, 252, 30, 0.5)', border: 'rgb(251, 252, 30)', segmentation: [251, 252, 30] },  // Yellow
  { box: 'rgba(255, 0, 0, 0.5)', border: 'rgb(255, 0, 0)', segmentation: [255, 0, 0] },           // Red
  { box: 'rgba(0, 255, 0, 0.5)', border: 'rgb(0, 255, 0)', segmentation: [0, 255, 0] },           // Green
  { box: 'rgba(0, 0, 255, 0.5)', border: 'rgb(0, 0, 255)', segmentation: [0, 0, 255] },           // Blue
  { box: 'rgba(255, 165, 0, 0.5)', border: 'rgb(255, 165, 0)', segmentation: [255, 165, 0] },     // Orange
  { box: 'rgba(128, 0, 128, 0.5)', border: 'rgb(128, 0, 128)', segmentation: [128, 0, 128] },     // Purple
  { box: 'rgba(255, 192, 203, 0.5)', border: 'rgb(255, 192, 203)', segmentation: [255, 192, 203] }, // Pink
  { box: 'rgba(0, 255, 255, 0.5)', border: 'rgb(0, 255, 255)', segmentation: [0, 255, 255] }      // Cyan
];

function initMedSAM() {
  console.log('Initializing MedSAM Tool (Simple Version)');
  
  // Get DOM elements
  uploadedImage = document.getElementById('uploadedImage');
  medsamCanvas = document.getElementById('medsamCanvas');
  
  if (!medsamCanvas) {
    console.error('MedSAM canvas not found, creating it');
    medsamCanvas = document.createElement('canvas');
    medsamCanvas.id = 'medsamCanvas';
    const zoomContainer = document.getElementById('zoomContainer');
    if (zoomContainer) {
      zoomContainer.appendChild(medsamCanvas);
    } else {
      console.error('Zoom container not found');
      return;
    }
  }
  
  medsamCtx = medsamCanvas.getContext('2d');
  
  // Position canvas over the image
  setupCanvas();
  
  // Wait for image to load if it hasn't already
  if (uploadedImage.complete) {
    onImageLoaded();
  } else {
    uploadedImage.onload = onImageLoaded;
  }
  
  // Add event listeners for canvas interactions
  medsamCanvas.addEventListener('mousedown', handleMouseDown);
  medsamCanvas.addEventListener('mousemove', handleMouseMove);
  medsamCanvas.addEventListener('mouseup', handleMouseUp);
  medsamCanvas.addEventListener('mouseleave', handleMouseLeave);
  
  // Add event listeners for buttons
  const undoBtn = document.getElementById('undoMedsamBtn');
  if (undoBtn) undoBtn.addEventListener('click', undoLastBox);
  
  const redoBtn = document.getElementById('redoMedsamBtn');
  if (redoBtn) redoBtn.addEventListener('click', redoLastBox);
  
  const resetBtn = document.getElementById('resetMedsamBtn');
  if (resetBtn) resetBtn.addEventListener('click', resetAllBoxes);
  
  console.log('MedSAM Tool initialized successfully');
}

function setupCanvas() {
  // Make canvas the same size as the image
  const rect = uploadedImage.getBoundingClientRect();
  medsamCanvas.width = rect.width;
  medsamCanvas.height = rect.height;
  
  // Position canvas directly over the image
  medsamCanvas.style.position = 'absolute';
  medsamCanvas.style.top = '0';
  medsamCanvas.style.left = '0';
  medsamCanvas.style.pointerEvents = 'auto';
  medsamCanvas.style.zIndex = '10';
  
  canvasSize = { width: medsamCanvas.width, height: medsamCanvas.height };
}

function onImageLoaded() {
  imageNaturalSize = {
    width: uploadedImage.naturalWidth,
    height: uploadedImage.naturalHeight
  };
  
  setupCanvas();
  console.log('MedSAM initialized with image:', imageNaturalSize);
}

function handleMouseDown(e) {
  console.log('Mouse down on MedSAM canvas');
  const rect = medsamCanvas.getBoundingClientRect();
  startX = e.clientX - rect.left;
  startY = e.clientY - rect.top;
  isDrawingMedsamBox = true;
  
  // Clear canvas and redraw existing content
  redrawAllBoxesAndSegmentation();
}

function handleMouseMove(e) {
  if (!isDrawingMedsamBox) return;
  
  const rect = medsamCanvas.getBoundingClientRect();
  const currentX = e.clientX - rect.left;
  const currentY = e.clientY - rect.top;
  
  // Clear canvas and redraw existing content
  redrawAllBoxesAndSegmentation();
  
  // Draw current box being drawn
  drawBox(startX, startY, currentX, currentY, '#73bbc5', true); // dashed for current drawing
}

function handleMouseUp(e) {
  if (!isDrawingMedsamBox) return;
  
  console.log('Mouse up - box completed');
  
  const rect = medsamCanvas.getBoundingClientRect();
  const endX = e.clientX - rect.left;
  const endY = e.clientY - rect.top;
  
  isDrawingMedsamBox = false;
  
  // Calculate bounding box
  const x1 = Math.min(startX, endX);
  const y1 = Math.min(startY, endY);
  const x2 = Math.max(startX, endX);
  const y2 = Math.max(startY, endY);
  
  // Check if box is too small
  if (x2 - x1 < 10 || y2 - y1 < 10) {
    console.log('Bounding box too small, ignoring');
    redrawAllBoxesAndSegmentation();
    return;
  }
  
  // Calculate the bounding box in terms of the original image
  const scaleX = imageNaturalSize.width / canvasSize.width;
  const scaleY = imageNaturalSize.height / canvasSize.height;
  
  const imageBox = [
    Math.round(x1 * scaleX),
    Math.round(y1 * scaleY),
    Math.round(x2 * scaleX),
    Math.round(y2 * scaleY)
  ];
  
  const canvasBox = [x1, y1, x2, y2];
  
  // Get color for this box
  const color = getNextBoxColor();
  
  // Add box to collection
  currentBoxes.push({
    canvas: canvasBox,
    image: imageBox,
    color: color
  });
  
  console.log(`Added box ${currentBoxes.length}: ${imageBox}`);
  
  // Perform segmentation for every new box
  performSegmentation(imageBox);
}

function handleMouseLeave(e) {
  if (isDrawingMedsamBox) {
    isDrawingMedsamBox = false;
    redrawAllBoxesAndSegmentation();
  }
}

function performSegmentation(imageBox) {
  console.log('Starting segmentation for box:', imageBox);
  showLoading(true);
  
  // Get the relative image path from the src attribute
  const imagePath = getImagePath();
  
  // Get the color for this box
  const color = currentBoxes[currentBoxes.length - 1].color;
  
  // Prepare request data
  const requestData = {
    image_path: imagePath,
    mode: 'replace', // Always use replace mode to send all boxes
    boxes: currentBoxes.map(box => box.image) // Send all boxes
  };
  
  console.log('Sending segmentation request:', requestData);
  
  // Call the simple API endpoint
  fetch('/api/medsam/segment', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-CSRFToken': getCSRFToken()
    },
    body: JSON.stringify(requestData)
  })
  .then(response => {
    console.log('Response received:', response.status);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    return response.json();
  })
  .then(data => {
    showLoading(false);
    console.log('Segmentation response:', data);
    
    if (data.success) {
      // Update the cumulative overlay image
      updateCumulativeSegmentation(data);
      
      // Update UI
      updateUI();
      updateStats(data);
      
      // Show success message if it's a fallback
      if (data.implementation === 'fallback_dummy') {
        console.warn('Using fallback segmentation - MedSAM model not available');
      }
    } else {
      console.error('Segmentation failed:', data.error);
      alert('Segmentation failed: ' + data.error);
      
      // Remove the box from our local array since segmentation failed
      currentBoxes.pop();
      redrawAllBoxesAndSegmentation();
    }
  })
  .catch(error => {
    showLoading(false);
    console.error('Error during segmentation:', error);
    alert('An error occurred during segmentation: ' + error.message);
    
    // Remove the box from our local array since segmentation failed
    currentBoxes.pop();
    redrawAllBoxesAndSegmentation();
  });
}

function updateCumulativeSegmentation(data) {
  console.log('Updating segmentation result');

  // Store the new overlay image
  const img = new Image();
  img.onload = function () {
    // Add the new segmentation result to our array
    segmentationResults.push({
      overlay: img,
      box: currentBoxes[currentBoxes.length - 1],
      stats: {
        mask_pixels: data.new_mask_pixels,
        total_boxes: data.total_boxes,
        total_mask_pixels: data.total_mask_pixels
      }
    });
    
    // Update the cumulative overlay with the latest result
    cumulativeOverlayImage = img;
    
    // Redraw everything with the new result
    redrawAllBoxesAndSegmentation();
    
    console.log('Segmentation updated successfully');
  };

  img.onerror = function () {
    console.error('Failed to load overlay image');
  };

  img.src = data.overlay_data;
}

function redrawAllBoxesAndSegmentation() {
  // Clear the canvas
  medsamCtx.clearRect(0, 0, medsamCanvas.width, medsamCanvas.height);
  
  // Draw all segmentation results
  segmentationResults.forEach(result => {
    medsamCtx.drawImage(result.overlay, 0, 0, medsamCanvas.width, medsamCanvas.height);
  });
  
  // Draw all boxes on top
  drawAllBoxesOnTop();
}

function drawBox(x1, y1, x2, y2, color, dashed = false) {
  const boxColor = typeof color === 'string' ? color : color.border;
  
  // Draw border
  medsamCtx.strokeStyle = boxColor;
  medsamCtx.lineWidth = 2;
  
  if (dashed) {
    medsamCtx.setLineDash([5, 5]);
  } else {
    medsamCtx.setLineDash([]);
  }
  
  medsamCtx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  
  // Draw corner points
  medsamCtx.setLineDash([]);
  drawPoint(x1, y1, boxColor);
  drawPoint(x2, y1, boxColor);
  drawPoint(x1, y2, boxColor);
  drawPoint(x2, y2, boxColor);
}

function drawPoint(x, y, color = 'red') {
  medsamCtx.fillStyle = color;
  medsamCtx.beginPath();
  medsamCtx.arc(x, y, 4, 0, Math.PI * 2);
  medsamCtx.fill();
}

function getNextBoxColor() {
  return colorPalette[currentBoxes.length % colorPalette.length];
}

function resetAllBoxes() {
  console.log('Resetting all boxes');
  
  // Clear local state
  currentBoxes = [];
  segmentationResults = [];
  cumulativeOverlayImage = null;
  pendingSegmentation = false;
  undoHistory = []; // Clear undo history

  // Clear canvas
  medsamCtx.clearRect(0, 0, medsamCanvas.width, medsamCanvas.height);
  
  // Update UI
  updateUI();
  updateStats({ total_boxes: 0, total_mask_pixels: 0 });
}

function undoLastBox() {
  if (currentBoxes.length > 0) {
    console.log('Removing last box');
    
    // Store the last box and its segmentation in history
    undoHistory.push({
      box: currentBoxes[currentBoxes.length - 1],
      segmentation: segmentationResults[segmentationResults.length - 1]
    });
    
    // Remove the last box and its segmentation result
    currentBoxes.pop();
    segmentationResults.pop();
    
    // Redraw remaining boxes and segmentations
    redrawAllBoxesAndSegmentation();
    updateUI();
    updateStats({ total_boxes: currentBoxes.length, total_mask_pixels: 0 });
    
    // Set flag to indicate we need to update segmentation
    pendingSegmentation = true;
  }
}

function redoLastBox() {
  if (undoHistory.length > 0) {
    console.log('Restoring last undone box');
    
    // Get the last undone box and its segmentation
    const lastUndone = undoHistory.pop();
    
    // Add back the box and its segmentation
    currentBoxes.push(lastUndone.box);
    segmentationResults.push(lastUndone.segmentation);
    
    // Redraw everything
    redrawAllBoxesAndSegmentation();
    updateUI();
    updateStats({ total_boxes: currentBoxes.length, total_mask_pixels: 0 });
    
    // Set flag to indicate we need to update segmentation
    pendingSegmentation = true;
  }
}

function drawAllBoxesOnTop() {
  currentBoxes.forEach((boxData, index) => {
    const [x1, y1, x2, y2] = boxData.canvas;
    drawBox(x1, y1, x2, y2, boxData.color, false);
    
    // Add box number label without background
    medsamCtx.fillStyle = boxData.color.border;
    medsamCtx.font = '12px Arial';
    medsamCtx.fillText(`${index + 1}`, x1 + 5, y1 + 15);
  });
}

function updateUI() {
  const undoBtn = document.getElementById('undoMedsamBtn');
  const redoBtn = document.getElementById('redoMedsamBtn');
  const resetBtn = document.getElementById('resetMedsamBtn');
  
  if (undoBtn) undoBtn.disabled = currentBoxes.length === 0;
  if (redoBtn) redoBtn.disabled = undoHistory.length === 0;
  if (resetBtn) resetBtn.disabled = currentBoxes.length === 0;
}

function updateStats(data) {
  const statsElement = document.getElementById('medsamStats');
  const boxCountElement = document.getElementById('boxCount');
  const segmentedAreasElement = document.getElementById('segmentedAreas');
  
  if (statsElement) {
    statsElement.style.display = 'block';
  }
  
  if (boxCountElement) {
    boxCountElement.textContent = `${data.total_boxes || currentBoxes.length} boxes drawn`;
  }
  
  if (segmentedAreasElement) {
    const pixels = data.total_mask_pixels || 0;
    segmentedAreasElement.textContent = `${pixels.toLocaleString()} pixels segmented`;
  }
}

function showLoading(show = true) {
  const loadingOverlay = document.getElementById('loadingOverlay');
  if (loadingOverlay) {
    loadingOverlay.style.display = show ? 'flex' : 'none';
  }
  
  // Also update button states during loading
  const undoBtn = document.getElementById('undoMedsamBtn');
  const redoBtn = document.getElementById('redoMedsamBtn');
  const resetBtn = document.getElementById('resetMedsamBtn');
  
  if (undoBtn) undoBtn.disabled = show || currentBoxes.length === 0;
  if (redoBtn) redoBtn.disabled = show || undoHistory.length === 0;
  if (resetBtn) resetBtn.disabled = show || currentBoxes.length === 0;
}

function getImagePath() {
  const imageSrc = uploadedImage.src;
  return imageSrc.split('/media/')[1];
}

function getCSRFToken() {
  // Try multiple methods to get CSRF token
  if (window.csrfToken) {
    return window.csrfToken;
  }
  
  const metaToken = document.querySelector('meta[name="csrf-token"]');
  if (metaToken) {
    return metaToken.getAttribute('content');
  }
  
  const cookieValue = document.cookie
    .split('; ')
    .find(row => row.startsWith('csrftoken='));
  
  if (cookieValue) {
    return cookieValue.split('=')[1];
  }
  
  console.warn('CSRF token not found');
  return '';
}

// Make function available globally
window.initMedSAM = initMedSAM;

// Auto-initialize if we're in MedSAM mode
document.addEventListener('DOMContentLoaded', function() {
  if (window.algorithm === 'medsam') {
    // Small delay to ensure DOM is fully ready
    setTimeout(initMedSAM, 100);
  }
});