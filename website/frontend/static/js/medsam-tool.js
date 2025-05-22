/**
 * MedSAM Tool Integration for KITE
 * Provides interactive segmentation using the Medical Segment Anything Model
 */

// Global variables
let medsamCanvas;
let medsamCtx;
let uploadedImage;
let isDrawing = false;
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
  
  const segmentBtn = document.getElementById('medsamSegmentBtn');
  if (segmentBtn) segmentBtn.addEventListener('click', runBatchSegmentation);
  
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
  medsamCanvas.style.zIndex = '2';
  
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
  const rect = medsamCanvas.getBoundingClientRect();
  startX = e.clientX - rect.left;
  startY = e.clientY - rect.top;
  isDrawing = true;
  
  // Clear canvas and redraw existing boxes
  redrawAllBoxes();
}

function handleMouseMove(e) {
  if (!isDrawing) return;
  
  const rect = medsamCanvas.getBoundingClientRect();
  const currentX = e.clientX - rect.left;
  const currentY = e.clientY - rect.top;
  
  // Clear canvas and redraw existing boxes
  redrawAllBoxes();
  
  // Draw current box being drawn
  drawBox(startX, startY, currentX, currentY, 'red', true); // dashed for current drawing
}

function handleMouseUp(e) {
  if (!isDrawing) return;
  
  const rect = medsamCanvas.getBoundingClientRect();
  const endX = e.clientX - rect.left;
  const endY = e.clientY - rect.top;
  
  isDrawing = false;
  
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
  
  // Redraw all boxes
  redrawAllBoxes();
  
  // Update UI
  updateUI();
  
  console.log(`Added box ${currentBoxes.length}: ${imageBox}`);
}

function handleMouseLeave(e) {
  if (isDrawing) {
    isDrawing = false;
    redrawAllBoxes();
  }
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

function redrawAllBoxes() {
  // Clear canvas
  medsamCtx.clearRect(0, 0, medsamCanvas.width, medsamCanvas.height);
  
  // Draw all existing boxes
  currentBoxes.forEach((boxData, index) => {
    const [x1, y1, x2, y2] = boxData.canvas;
    drawBox(x1, y1, x2, y2, boxData.color, false);
    
    // Add box number
    medsamCtx.fillStyle = boxData.color;
    medsamCtx.font = '14px Arial';
    medsamCtx.fillText(`${index + 1}`, x1 + 5, y1 + 15);
  });
}

function undoLastBox() {
  if (currentBoxes.length > 0) {
    currentBoxes.pop();
    redrawAllBoxes();
    updateUI();
    console.log(`Removed last box. Remaining: ${currentBoxes.length}`);
  }
}

function resetAllBoxes() {
  currentBoxes = [];
  segmentationHistory = [];
  currentSegmentation = null;
  
  // Clear canvas
  medsamCtx.clearRect(0, 0, medsamCanvas.width, medsamCanvas.height);
  
  updateUI();
  console.log('Reset all boxes and segmentations');
}

function updateUI() {
  const undoBtn = document.getElementById('undoMedsamBtn');
  const resetBtn = document.getElementById('resetMedsamBtn');
  const segmentBtn = document.getElementById('medsamSegmentBtn');
  
  if (undoBtn) undoBtn.disabled = currentBoxes.length === 0;
  if (resetBtn) resetBtn.disabled = currentBoxes.length === 0;
  if (segmentBtn) {
    segmentBtn.disabled = currentBoxes.length === 0;
    segmentBtn.textContent = currentBoxes.length > 1 ? 
      `Run Batch Segmentation (${currentBoxes.length} areas)` : 
      'Run MedSAM Segmentation';
  }
}

function showLoading(show = true) {
  const loadingOverlay = document.getElementById('loadingOverlay');
  if (loadingOverlay) {
    loadingOverlay.style.display = show ? 'flex' : 'none';
  }
}

function runBatchSegmentation() {
  if (currentBoxes.length === 0) {
    alert('Please draw at least one bounding box first.');
    return;
  }
  
  showLoading(true);
  
  // Get the relative image path from the src attribute
  const imageSrc = uploadedImage.src;
  const imagePath = imageSrc.split('/media/')[1];
  
  // Prepare request data
  const requestData = {
    image_path: imagePath
  };
  
  // Add boxes (use batch format even for single box for consistency)
  if (currentBoxes.length === 1) {
    requestData.box = currentBoxes[0].image;
  } else {
    requestData.boxes = currentBoxes.map(boxData => boxData.image);
  }
  
  console.log('Running segmentation with data:', requestData);
  
  // Call the backend API
  fetch('/api/medsam/segment', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-CSRFToken': window.csrfToken
    },
    body: JSON.stringify(requestData)
  })
  .then(response => response.json())
  .then(data => {
    showLoading(false);
    
    if (data.success) {
      handleSegmentationSuccess(data);
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

function handleSegmentationSuccess(data) {
  console.log('Segmentation completed successfully:', data);
  
  // Store segmentation result
  currentSegmentation = {
    overlayPath: data.overlay_path,
    maskPath: data.mask_path,
    overlayData: data.overlay_data,
    boxesProcessed: data.boxes_processed,
    totalPixels: data.total_mask_pixels,
    isBatch: data.is_batch,
    individualResults: data.individual_results || []
  };
  
  // Add to history
  segmentationHistory.push(currentSegmentation);
  
  // Display the segmentation overlay
  displaySegmentationResult();
  
  // Show summary
  showSegmentationSummary(data);
}

function displaySegmentationResult() {
  if (!currentSegmentation) return;
  
  // Create overlay image
  const img = new Image();
  img.onload = function() {
    // Clear canvas and draw the segmentation result
    medsamCtx.clearRect(0, 0, medsamCanvas.width, medsamCanvas.height);
    medsamCtx.drawImage(img, 0, 0, medsamCanvas.width, medsamCanvas.height);
    
    // Draw boxes on top of segmentation
    currentBoxes.forEach((boxData, index) => {
      const [x1, y1, x2, y2] = boxData.canvas;
      drawBox(x1, y1, x2, y2, 'white', false); // White boxes on segmentation
      
      // Add box number
      medsamCtx.fillStyle = 'white';
      medsamCtx.font = '14px Arial';
      medsamCtx.strokeStyle = 'black';
      medsamCtx.lineWidth = 1;
      medsamCtx.strokeText(`${index + 1}`, x1 + 5, y1 + 15);
      medsamCtx.fillText(`${index + 1}`, x1 + 5, y1 + 15);
    });
  };
  
  img.src = currentSegmentation.overlayData;
}

function showSegmentationSummary(data) {
  let summaryText = `Segmentation completed!\n`;
  summaryText += `• Areas processed: ${data.boxes_processed}\n`;
  summaryText += `• Total segmented pixels: ${data.total_mask_pixels.toLocaleString()}\n`;
  
  if (data.individual_results && data.individual_results.length > 1) {
    summaryText += `\nArea breakdown:\n`;
    data.individual_results.forEach((result, index) => {
      summaryText += `• Area ${index + 1}: ${result.mask_pixels.toLocaleString()} pixels\n`;
    });
  }
  
  console.log(summaryText);
  
  // You could also show this in a modal or notification instead of alert
  // For now, just log it and proceed to show final result
  showFinalResult();
}

function showFinalResult() {
  if (!currentSegmentation) return;
  
  // Display the result in the main result section
  const resultImage = document.getElementById('segmentedResultImage');
  if (resultImage) {
    resultImage.src = currentSegmentation.overlayData;
    
    // Show the result section
    const resultSection = document.getElementById('segmentationResult');
    if (resultSection) {
      resultSection.style.display = 'block';
    }
    
    // Set up download link
    const downloadLink = document.getElementById('downloadSegmentedImage');
    if (downloadLink) {
      downloadLink.href = currentSegmentation.maskPath;
      downloadLink.download = 'medsam_segmented_' + window.imageName;
      downloadLink.style.display = 'inline-block';
    }
    
    // Scroll to results
    if (resultSection) {
      resultSection.scrollIntoView({ behavior: 'smooth' });
    }
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