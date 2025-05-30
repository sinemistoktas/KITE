// website/frontend/static/js/main.js

import { state, initializeFromServer } from './state.js';
import { bindUIEvents, updateMethodDescription, initializeAnnotationsFromPredictions } from './events.js';
import { redrawAnnotations } from './canvas-tools.js';
import { initColorPicker } from './color-picker.js';
import {
    handleAnnotations,
    handleAnnotationsWithResultLoading,
    loadSegmentationResultsAsAnnotations,
    handlePreprocessedImg,
    loadAnnotations,
    downloadAnnotations
} from './api-service.js';
import { initBoxTool } from './box-tool.js';

// Configuration: Set your desired display size
const DESIRED_MAX_WIDTH = 800;  // Target width
const DESIRED_MAX_HEIGHT = 600; // Target height

document.addEventListener('DOMContentLoaded', () => {
    const annotationCanvas = document.getElementById("annotationCanvas");
    const predictionCanvas = document.getElementById("predictionCanvas");
    const medsamCanvas = document.getElementById("medsamCanvas");
    const img = document.getElementById("uploadedImage");

    // Only proceed if the necessary elements are present.
    if (!annotationCanvas || !predictionCanvas || !img) {
        console.log('📷 No image uploaded yet, skipping canvas setup');
        console.log('Canvas elements found:', {
            annotationCanvas: !!annotationCanvas,
            predictionCanvas: !!predictionCanvas,
            medsamCanvas: !!medsamCanvas,
            img: !!img
        });
        return;
    }

    // Store references in global state
    state.annotationCanvas = annotationCanvas;
    state.predictionCanvas = predictionCanvas;
    state.annotationCtx = annotationCanvas.getContext("2d");
    state.predictionCtx = predictionCanvas.getContext("2d");
    state.imageName = window.imageName;

    // Initialize scale tracking
    state.displayScale = 1;
    state.originalImageNaturalDimensions = { width: 0, height: 0 };

    // Handle both new algorithm system and legacy segmentation method system
    if (window.algorithm) {
        state.algorithm = window.algorithm;
        state.unetMode = window.algorithm === 'U-Net';
        state.medsamMode = window.algorithm === 'MedSAM';
        state.kiteMode = window.algorithm === 'KITE';
    }

    if (window.segmentationMethod) {
        state.segmentationMethod = window.segmentationMethod;
        state.unetMode = window.segmentationMethod === 'unet';
    }

    // Initialize from server data if available
    if (typeof initializeFromServer === 'function') {
        initializeFromServer({
            imageName: window.imageName,
            algorithm: window.algorithm,
            segmentationMethod: window.segmentationMethod,
            predictedPoints: window.predictedPoints
        });
    }

    // Handle predicted points for UNet auto-processing
    if (window.predictedPoints && window.predictedPoints.length > 0) {
        try {
            const predictions = [{ points: window.predictedPoints }];
            initializeAnnotationsFromPredictions(predictions);
        } catch (error) {
            console.error('Error initializing UNet predictions:', error);
        }
    }

    // ✅ Advanced canvas setup with scale tracking and coordinate conversion
    const resizeCanvasToImage = () => {
        // Wait for image to be fully loaded
        if (!img.complete) {
            img.onload = resizeCanvasToImage;
            return;
        }

        // Store original natural dimensions
        const naturalWidth = img.naturalWidth;
        const naturalHeight = img.naturalHeight;
        state.originalImageNaturalDimensions = { width: naturalWidth, height: naturalHeight };

        console.log(`Original image size: ${naturalWidth}x${naturalHeight}`);

        // Calculate scaling to fit within desired dimensions while maintaining aspect ratio
        const scaleX = DESIRED_MAX_WIDTH / naturalWidth;
        const scaleY = DESIRED_MAX_HEIGHT / naturalHeight;

        // Use the smaller scale to ensure image fits within bounds, but ensure minimum scale
        const scale = Math.min(scaleX, scaleY);
        const finalScale = Math.max(scale, 1.2); // At least 1.2x bigger

        // Store the scale factor for coordinate conversion
        state.displayScale = finalScale;

        // Calculate final display dimensions
        const displayWidth = Math.round(naturalWidth * finalScale);
        const displayHeight = Math.round(naturalHeight * finalScale);

        console.log(`Scaling by ${finalScale.toFixed(2)}x to ${displayWidth}x${displayHeight}`);

        // Store dimensions in state
        state.originalImageDimensions = { width: displayWidth, height: displayHeight };
        state.viewportCenterX = displayWidth / 2;
        state.viewportCenterY = displayHeight / 2;

        // Get the zoom container
        const zoomContainer = document.getElementById("zoomContainer");

        // Reset any existing transforms first
        img.style.transform = '';
        img.style.transformOrigin = '';
        img.style.position = 'relative';
        img.style.left = '';
        img.style.top = '';

        // Resize image
        img.style.width = `${displayWidth}px`;
        img.style.height = `${displayHeight}px`;

        // Set zoom container size to match image
        if (zoomContainer) {
            zoomContainer.style.width = `${displayWidth}px`;
            zoomContainer.style.height = `${displayHeight}px`;
            zoomContainer.style.position = 'relative';
        }

        // Resize and position all canvases to exactly match the image
        [annotationCanvas, predictionCanvas, medsamCanvas].forEach(canvas => {
            if (canvas) {
                // Set canvas internal dimensions
                canvas.width = displayWidth;
                canvas.height = displayHeight;

                // Set canvas CSS dimensions to match
                canvas.style.width = `${displayWidth}px`;
                canvas.style.height = `${displayHeight}px`;

                // Position canvas exactly over the image
                canvas.style.position = "absolute";
                canvas.style.top = "0px";
                canvas.style.left = "0px";
                canvas.style.pointerEvents = "auto";

                // Reset any transforms
                canvas.style.transform = "";
                canvas.style.transformOrigin = "";
            }
        });

        // Set specific pointer events for canvases
        annotationCanvas.style.pointerEvents = "auto";
        predictionCanvas.style.pointerEvents = "none";

        // Redraw annotations after everything is positioned
        setTimeout(() => {
            redrawAnnotations();
        }, 50);

        console.log(`Canvas alignment completed: ${displayWidth}x${displayHeight}, scale: ${finalScale}`);
    };

    // Function to convert display coordinates to original image coordinates
    window.convertDisplayToOriginal = function(displayCoords) {
        if (!state.displayScale || !state.originalImageNaturalDimensions) {
            return displayCoords;
        }

        return displayCoords.map(coord => ({
            x: Math.round(coord.x / state.displayScale),
            y: Math.round(coord.y / state.displayScale)
        }));
    };

    // Function to convert original coordinates to display coordinates
    window.convertOriginalToDisplay = function(originalCoords) {
        if (!state.displayScale) {
            return originalCoords;
        }

        return originalCoords.map(coord => ({
            x: Math.round(coord.x * state.displayScale),
            y: Math.round(coord.y * state.displayScale)
        }));
    };

   window.handleAnnotations = function() {
    // Check if we have manual annotations that need coordinate conversion
    const hasManualAnnotations = state.scribbles && state.scribbles.filter(s => !s.isPrediction).length > 0;

    if (hasManualAnnotations && state.displayScale && state.displayScale !== 1) {
        console.log('Converting manual annotations from display scale to backend scale');
        const originalRedrawFunction = window.redrawAnnotations || redrawAnnotations;
        const originalScribbles = JSON.parse(JSON.stringify(state.scribbles));
        
        const backendScaleAnnotations = state.scribbles
            .filter(s => !s.isPrediction)
            .map(stroke => ({
                ...stroke,
                points: stroke.points.map(point => ({
                    x: Math.round(point.x / state.displayScale),
                    y: Math.round(point.y / state.displayScale)
                }))
            }));

        // Temporarily replace the scribbles with original scale coordinates
        const predictionsOnly = state.scribbles.filter(s => s.isPrediction);
        state.coordinatesAlreadyScaled = true;
        
        // Create temporary state for API call.
        const tempScribbles = [...predictionsOnly, ...backendScaleAnnotations];
        const originalStateScribbles = state.scribbles;
        
        // Temporarily set state for API call.
        state.scribbles = tempScribbles;

        // Call API.
        const apiCall = typeof handleAnnotationsWithResultLoading === 'function' 
            ? handleAnnotationsWithResultLoading() 
            : handleAnnotations();

        // ✅ Restore everything immediately without triggering redraws
        state.scribbles = originalStateScribbles;
        state.coordinatesAlreadyScaled = false;

        return apiCall;
    } else {
        // No coordinate conversion needed - call original function directly
        return typeof handleAnnotationsWithResultLoading === 'function' 
            ? handleAnnotationsWithResultLoading() 
            : handleAnnotations();
    }
};

    // Initial resize
    resizeCanvasToImage();

    // Handle window resize events
    window.addEventListener("resize", resizeCanvasToImage);

    // Hook up all API functions
    window.handleAnnotationsWithResultLoading = handleAnnotationsWithResultLoading;
    window.handleAnnotationsBasic = handleAnnotations;
    window.loadSegmentationResultsAsAnnotations = loadSegmentationResultsAsAnnotations;
    window.handlePreprocessedImg = handlePreprocessedImg;
    window.loadAnnotations = loadAnnotations;
    window.downloadAnnotations = downloadAnnotations;

    if (window.algorithm) {
        const normalizedAlgorithm = window.algorithm.toLowerCase();
        state.algorithm = normalizedAlgorithm;
        state.unetMode = normalizedAlgorithm === 'unet';
        state.medsamMode = normalizedAlgorithm === 'medsam';
        state.kiteMode = normalizedAlgorithm === 'kite';
    }

    // Initialize UI components
    bindUIEvents();
    initColorPicker();
    initBoxTool();

    // Update method description (for legacy radio button system)
    if (typeof updateMethodDescription === 'function') {
        updateMethodDescription();
    }

    // Adjust UI for selected method
    const segmentBtn = document.querySelector('.ready-segment-btn');
    if (segmentBtn) {
        if (state.unetMode) {
            segmentBtn.innerHTML = '<i class="fa-solid fa-hurricane me-2"></i> Ready to Segment with UNet!';
        } else if (state.medsamMode) {
            segmentBtn.innerHTML = '<i class="fa-solid fa-hurricane me-2"></i> Ready to Segment with MedSAM!';
        } else if (state.kiteMode || (!state.algorithm && !state.segmentationMethod)) {
            // Default KITE mode or no specific algorithm selected
            segmentBtn.innerHTML = '<i class="fa-solid fa-hurricane me-2"></i> Ready to Segment!';
        }
    }

    console.log('KITE segmentation tool initialized with coordinate conversion and enhanced functionality');
});

// Debug function
window.testFunction = function() {
    alert('JavaScript is working!');
};

// Load last segmentation results as annotations
window.loadLastSegmentationAsAnnotations = function() {
    if (window.lastSegmentationData && window.lastSegmentationData.final_mask) {
        loadSegmentationResultsAsAnnotations(window.lastSegmentationData);
        alert('Segmentation results loaded as editable annotations!');
    } else {
        alert('No segmentation results available to load. Please run segmentation first.');
    }
};