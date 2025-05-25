// website/frontend/static/js/medsam-tool.js

/**
 * MedSAM Tool Integration for KITE
 * Provides interactive segmentation using the Medical Segment Anything Model
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
let segmentationHistory = [];
let currentBoxes = []; // Store all drawn boxes
let currentSegmentation = null;

function initMedSAM() {
  console.log('Initializing MedSAM Tool');
  
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
  
  const resetBtn = document.getElementById('resetMedsamBtn');
  if (resetBtn) resetBtn.addEventListener('click', resetAllBoxes);
  
  // Handle zoom changes
  const zoomInBtn = document.getElementById('zoomInBtn');
  if (zoomInBtn) zoomInBtn.addEventListener('click', updateCanvasOnZoom);
  
  const zoomOutBtn = document.getElementById('zoomOutBtn');
  if (zoomOutBtn) zoomOutBtn.addEventListener('click', updateCanvasOnZoom);
  
  const resetZoomBtn = document.getElementById('resetZoomBtn');
  if (resetZoomBtn) resetZoomBtn.addEventListener('click', updateCanvasOnZoom);
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
  
  // Position the canvas relative to the container
  const container = document.getElementById('zoomContainer');
  if (container) {
    container.style.position = 'relative';
  }
  
  canvasSize = { width: medsamCanvas.width, height: medsamCanvas.height };
}

function updateCanvasOnZoom() {
  setTimeout(() => {
    setupCanvas();
    redrawAllBoxes();
  }, 100);
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
  
  // Clear canvas and redraw existing boxes
  redrawAllBoxes();
}

function handleMouseMove(e) {
  if (!isDrawingMedsamBox) return;
  
  const rect = medsamCanvas.getBoundingClientRect();
  const currentX = e.clientX - rect.left;
  const currentY = e.clientY - rect.top;
  
  // Clear canvas and redraw existing boxes
  redrawAllBoxes();
  
  // Draw current box being drawn
  drawBox(startX, startY, currentX, currentY, 'red', true); // dashed for current drawing
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
    redrawAllBoxes();
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
  
  // Add box to collection
  currentBoxes.push({
    canvas: canvasBox,
    image: imageBox,
    color: getNextBoxColor()
  });
  
  // Update UI
  updateUI();
  
  console.log(`Added box ${currentBoxes.length}: ${imageBox}`);
  
  // OTOMATIK SEGMENTASYON BAŞLAT
  performAutomaticSegmentation(imageBox, canvasBox);
}

function handleMouseLeave(e) {
  if (isDrawingMedsamBox) {
    isDrawingMedsamBox = false;
    redrawAllBoxes();
  }
}

function performAutomaticSegmentation(imageBox, canvasBox) {
  console.log('Starting automatic segmentation for box:', imageBox);
  showLoading(true);
  
  // Get the relative image path from the src attribute
  const imageSrc = uploadedImage.src;
  const imagePath = imageSrc.split('/media/')[1];
  
  // Prepare request data for single box
  const requestData = {
    image_path: imagePath,
    box: imageBox
  };
  
  console.log('Sending segmentation request:', requestData);
  
  // Call the backend API
  fetch('/api/medsam/segment', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-CSRFToken': window.csrfToken
    },
    body: JSON.stringify(requestData)
  })
  .then(response => {
    console.log('Response received:', response.status);
    return response.json();
  })
  .then(data => {
    showLoading(false);
    console.log('Segmentation response:', data);
    
    if (data.success) {
      // Update the canvas with segmentation result immediately
      updateCanvasWithSegmentation(data);
    } else {
      console.error('Segmentation failed:', data.error);
      alert('Segmentation failed: ' + data.error);
    }
  })
  .catch(error => {
    showLoading(false);
    console.error('Error during segmentation:', error);
    alert('An error occurred during segmentation. Please try again.');
  });
}

function updateCanvasWithSegmentation(data) {
  console.log('Updating canvas with segmentation result');
  
  // Store segmentation result
  currentSegmentation = {
    overlayPath: data.overlay_path,
    maskPath: data.mask_path,
    overlayData: data.overlay_data,
    boxesProcessed: data.boxes_processed,
    totalPixels: data.total_mask_pixels
  };
  
  // Create overlay image
  const img = new Image();
  img.onload = function() {
    console.log('Overlay image loaded successfully');
    
    // Clear canvas and draw the segmentation result
    medsamCtx.clearRect(0, 0, medsamCanvas.width, medsamCanvas.height);
    medsamCtx.drawImage(img, 0, 0, medsamCanvas.width, medsamCanvas.height);
    
    // Draw boxes on top
    drawAllBoxesOnTop();
    
    console.log('Segmentation overlay applied to canvas');
  };
  
  img.onerror = function() {
    console.error('Failed to load segmentation overlay image');
  };
  
  img.src = data.overlay_data;
}


function drawBox(x1, y1, x2, y2, color = 'red', dashed = false) {
  medsamCtx.strokeStyle = color;
  medsamCtx.lineWidth = 2;
  
  if (dashed) {
    medsamCtx.setLineDash([5, 5]);
  } else {
    medsamCtx.setLineDash([]);
  }
  
  medsamCtx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  
  // Draw corner points
  medsamCtx.setLineDash([]);
  drawPoint(x1, y1, color);
  drawPoint(x2, y1, color);
  drawPoint(x1, y2, color);
  drawPoint(x2, y2, color);
}

function drawPoint(x, y, color = 'red') {
  medsamCtx.fillStyle = color;
  medsamCtx.beginPath();
  medsamCtx.arc(x, y, 4, 0, Math.PI * 2);
  medsamCtx.fill();
}

function getNextBoxColor() {
  const colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'pink', 'cyan'];
  return colors[currentBoxes.length % colors.length];
}


function resetAllBoxes() {
  currentBoxes = [];
  segmentationHistory = [];
  currentSegmentation = null; // reset segmentation
  
  // clear Canvas
  medsamCtx.clearRect(0, 0, medsamCanvas.width, medsamCanvas.height);
  
  updateUI();
  console.log('Reset all boxes and segmentations - canvas cleared completely');
}

function undoLastBox() {
  if (currentBoxes.length > 0) {
    // Remove the last box from the collection
    const removedBox = currentBoxes.pop();
    console.log(`Removed box ${currentBoxes.length + 1}:`, removedBox.image);
    
    // If we still have boxes, we need to re-run segmentation for remaining boxes
    if (currentBoxes.length > 0) {
      console.log('Re-running segmentation for remaining boxes');
      rerunSegmentationForAllBoxes();
    } else {
      // No boxes left - clear everything including segmentation
      console.log('No boxes remaining - clearing segmentation');
      currentSegmentation = null;
      segmentationHistory = [];
      
      // Clear canvas completely
      medsamCtx.clearRect(0, 0, medsamCanvas.width, medsamCanvas.height);
    }
    
    // Update UI buttons
    updateUI();
    console.log(`Remaining boxes: ${currentBoxes.length}`);
  }
}

// New helper function to re-run segmentation for all remaining boxes
function rerunSegmentationForAllBoxes() {
  if (currentBoxes.length === 0) {
    currentSegmentation = null;
    medsamCtx.clearRect(0, 0, medsamCanvas.width, medsamCanvas.height);
    return;
  }
  
  console.log('Re-running batch segmentation for', currentBoxes.length, 'boxes');
  showLoading(true);
  
  // Get the relative image path from the src attribute
  const imageSrc = uploadedImage.src;
  const imagePath = imageSrc.split('/media/')[1];
  
  // Prepare request data for all remaining boxes
  const requestData = {
    image_path: imagePath
  };
  
  // Use batch format if multiple boxes, single format if one box
  if (currentBoxes.length === 1) {
    requestData.box = currentBoxes[0].image;
  } else {
    requestData.boxes = currentBoxes.map(boxData => boxData.image);
  }
  
  console.log('Re-running segmentation with data:', requestData);
  
  // Call the backend API
  fetch('/api/medsam/segment', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-CSRFToken': window.csrfToken
    },
    body: JSON.stringify(requestData)
  })
  .then(response => {
    console.log('Re-segmentation response received:', response.status);
    return response.json();
  })
  .then(data => {
    showLoading(false);
    console.log('Re-segmentation response:', data);
    
    if (data.success) {
      // Update canvas with new segmentation result
      updateCanvasWithSegmentation(data);
    } else {
      console.error('Re-segmentation failed:', data.error);
      // Clear segmentation on failure
      currentSegmentation = null;
      redrawAllBoxes();
    }
  })
  .catch(error => {
    showLoading(false);
    console.error('Error during re-segmentation:', error);
    // Clear segmentation on error
    currentSegmentation = null;
    redrawAllBoxes();
  });
}

// Updated redrawAllBoxes function with better comments
function redrawAllBoxes() {
  // If we have segmentation data, redraw it first as background
  if (currentSegmentation) {
    console.log('Redrawing with segmentation background');
    const img = new Image();
    img.onload = function() {
      // Clear canvas completely
      medsamCtx.clearRect(0, 0, medsamCanvas.width, medsamCanvas.height);
      
      // Draw segmentation result as background
      medsamCtx.drawImage(img, 0, 0, medsamCanvas.width, medsamCanvas.height);
      
      // Draw all boxes on top of segmentation
      drawAllBoxesOnTop();
    };
    img.src = currentSegmentation.overlayData;
  } else {
    // No segmentation - just draw boxes on clean canvas
    console.log('Redrawing boxes only (no segmentation)');
    medsamCtx.clearRect(0, 0, medsamCanvas.width, medsamCanvas.height);
    drawAllBoxesOnTop();
  }
}

// Helper function to draw all boxes on top of whatever is already on canvas
function drawAllBoxesOnTop() {
  currentBoxes.forEach((boxData, index) => {
    const [x1, y1, x2, y2] = boxData.canvas;
    drawBox(x1, y1, x2, y2, boxData.color, false);
    
    // Add box number label
    medsamCtx.fillStyle = boxData.color;
    medsamCtx.font = '14px Arial';
    medsamCtx.fillText(`${index + 1}`, x1 + 5, y1 + 15);
  });
}

function updateUI() {
  const undoBtn = document.getElementById('undoMedsamBtn');
  const resetBtn = document.getElementById('resetMedsamBtn');
  
  if (undoBtn) undoBtn.disabled = currentBoxes.length === 0;
  if (resetBtn) resetBtn.disabled = currentBoxes.length === 0;
}

function showLoading(show = true) {
  const loadingOverlay = document.getElementById('loadingOverlay');
  if (loadingOverlay) {
    loadingOverlay.style.display = show ? 'flex' : 'none';
  }
}

// Make function available globally
window.initMedSAM = initMedSAM;

// Auto-initialize if we're in MedSAM mode
document.addEventListener('DOMContentLoaded', function() {
  if (window.algorithm === 'MedSAM') {
    // Small delay to ensure DOM is fully ready
    setTimeout(initMedSAM, 100);
  }
});