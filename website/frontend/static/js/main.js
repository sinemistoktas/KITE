// main.js
import { state, initializeFromServer } from './state.js';
import { bindUIEvents, updateMethodDescription, initializeAnnotationsFromPredictions } from './events.js';
import { redrawAnnotations } from './canvas-tools.js';
import { initColorPicker } from './color-picker.js';
import { handleAnnotations, handlePreprocessedImg } from './api-service.js';
import { initBoxTool } from './box-tool.js';

document.addEventListener('DOMContentLoaded', () => {
    const annotationCanvas = document.getElementById("annotationCanvas");
    const predictionCanvas = document.getElementById("predictionCanvas");
    const img = document.getElementById("uploadedImage");

    state.annotationCanvas = annotationCanvas;
    state.predictionCanvas = predictionCanvas;
    state.annotationCtx = annotationCanvas.getContext("2d");
    state.predictionCtx = predictionCanvas.getContext("2d");
    state.imageName = window.imageName;

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

    const resizeCanvasToImage = () => {
        const rect = img.getBoundingClientRect();
        state.originalImageDimensions = { width: rect.width, height: rect.height };
        state.viewportCenterX = rect.width / 2;
        state.viewportCenterY = rect.height / 2;

        [annotationCanvas, predictionCanvas].forEach(c => {
            c.width = rect.width;
            c.height = rect.height;
            c.style.width = `${rect.width}px`;
            c.style.height = `${rect.height}px`;
            c.style.position = "absolute";
            c.style.top = "0";
            c.style.left = "0";
        });

        annotationCanvas.style.pointerEvents = "auto";
        predictionCanvas.style.pointerEvents = "none";

        const zoomContainer = document.getElementById("zoomContainer");
        if (zoomContainer) {
            zoomContainer.style.width = `${rect.width}px`;
            zoomContainer.style.height = `${rect.height}px`;
        }

        redrawAnnotations();
    };

    if (img.complete) {
        resizeCanvasToImage();
    } else {
        img.onload = resizeCanvasToImage;
    }

    window.addEventListener("resize", resizeCanvasToImage);
    window.handleAnnotations = handleAnnotations;
    window.handlePreprocessedImg = handlePreprocessedImg;

    // Hook up all button + canvas events
    bindUIEvents();
    initColorPicker();
    initBoxTool();

    // Update method description (for legacy radio button system)
    if (typeof updateMethodDescription === 'function') {
        updateMethodDescription();
    }

    // Update UI based on selected algorithm/method
    if (state.unetMode) {
        const segmentBtn = document.querySelector('.ready-segment-btn');
        if (segmentBtn) {
            segmentBtn.innerHTML = '<i class="fa-solid fa-hurricane me-2"></i> Ready to Segment with UNet!';
        }
    }

    // Handle algorithm-specific UI updates
    if (state.medsamMode) {
        const segmentBtn = document.querySelector('.ready-segment-btn');
        if (segmentBtn) {
            segmentBtn.innerHTML = '<i class="fa-solid fa-hurricane me-2"></i> Ready to Segment with MedSAM!';
        }
    }

    if (state.kiteMode || (!state.algorithm && !state.segmentationMethod)) {
        // Default KITE mode or no specific algorithm selected
        const segmentBtn = document.querySelector('.ready-segment-btn');
        if (segmentBtn) {
            segmentBtn.innerHTML = '<i class="fa-solid fa-hurricane me-2"></i> Ready to Segment!';
        }
    }
});

// Debug function
window.testFunction = function() {
    alert('JavaScript is working!');
};