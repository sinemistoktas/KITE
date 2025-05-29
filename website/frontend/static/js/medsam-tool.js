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
let cumulativeOverlayImage = null; // Store the cumulative overlay

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
  
  // Add box to collection
  currentBoxes.push({
    canvas: canvasBox,
    image: imageBox,
    color: getNextBoxColor()
  });
  
  console.log(`Added box ${currentBoxes.length}: ${imageBox}`);
  
  // Perform segmentation
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
  
  // Prepare request data
  const requestData = {
    image_path: imagePath,
    box: imageBox,
    mode: 'add'
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
  
  // Draw the segmentation result if available
  if (cumulativeOverlayImage) {
    medsamCtx.drawImage(cumulativeOverlayImage, 0, 0, medsamCanvas.width, medsamCanvas.height);
  }
  
  // Draw all boxes on top
  drawAllBoxesOnTop();
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
  console.log('Resetting all boxes');
  
  // Clear local state
  currentBoxes = [];
  cumulativeOverlayImage = null;

  // Clear canvas
  medsamCtx.clearRect(0, 0, medsamCanvas.width, medsamCanvas.height);
  
  // Update UI
  updateUI();
  updateStats({ total_boxes: 0, total_mask_pixels: 0 });
}

function undoLastBox() {
  if (currentBoxes.length > 0) {
    console.log('Removing last box');
    currentBoxes.pop();
    
    if (currentBoxes.length === 0) {
      resetAllBoxes();
    } else {
      // For simplicity, just redraw without the last segmentation
      // In a more advanced version, you'd re-run segmentation for remaining boxes
      redrawAllBoxesAndSegmentation();
      updateUI();
    }
  }
}

function drawAllBoxesOnTop() {
  currentBoxes.forEach((boxData, index) => {
    const [x1, y1, x2, y2] = boxData.canvas;
    drawBox(x1, y1, x2, y2, boxData.color, false);
    
    // Add box number label with background for better visibility
    medsamCtx.fillStyle = 'rgba(255, 255, 255, 0.8)';
    medsamCtx.fillRect(x1 + 2, y1 + 2, 20, 16);
    
    medsamCtx.fillStyle = boxData.color;
    medsamCtx.font = '12px Arial';
    medsamCtx.fillText(`${index + 1}`, x1 + 5, y1 + 15);
  });
}

function updateUI() {
  const undoBtn = document.getElementById('undoMedsamBtn');
  const resetBtn = document.getElementById('resetMedsamBtn');
  
  if (undoBtn) undoBtn.disabled = currentBoxes.length === 0;
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
  const resetBtn = document.getElementById('resetMedsamBtn');
  
  if (undoBtn) undoBtn.disabled = show || currentBoxes.length === 0;
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