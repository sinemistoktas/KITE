// main.js
import { state, initializeFromServer } from './state.js';
import { bindUIEvents, updateMethodDescription, initializeAnnotationsFromPredictions } from './events.js';
import { redrawAnnotations } from './canvas-tools.js';
import { initColorPicker } from './color-picker.js';
import { handleAnnotations, handlePreprocessedImg, loadAnnotations, downloadAnnotations } from './api-service.js';
import { initBoxTool } from './box-tool.js';

document.addEventListener('DOMContentLoaded', () => {
    const annotationCanvas = document.getElementById("annotationCanvas");
    const predictionCanvas = document.getElementById("predictionCanvas");
    const medsamCanvas = document.getElementById("medsamCanvas");
    const img = document.getElementById("uploadedImage");

    // Only proceed if the necessary elements are present.
    if (!annotationCanvas || !predictionCanvas || !img) {
        console.log('📷 No image uploaded yet, skipping canvas setup');
        return;
    }

    // Store references in global state
    state.annotationCanvas = annotationCanvas;
    state.predictionCanvas = predictionCanvas;
    state.annotationCtx = annotationCanvas.getContext("2d");
    state.predictionCtx = predictionCanvas.getContext("2d");
    state.imageName = window.imageName;

    // Detect algorithm/method
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

    // Load server-passed predictions (e.g. UNet)
    if (typeof initializeFromServer === 'function') {
        initializeFromServer({
            imageName: window.imageName,
            algorithm: window.algorithm,
            segmentationMethod: window.segmentationMethod,
            predictedPoints: window.predictedPoints
        });
    }

    if (window.predictedPoints && window.predictedPoints.length > 0) {
        try {
            const predictions = [{ points: window.predictedPoints }];
            initializeAnnotationsFromPredictions(predictions);
        } catch (error) {
            console.error('Error initializing UNet predictions:', error);
        }
    }

    // ✅ Resize canvas to match displayed image
    const resizeCanvasToImage = () => {
        const imageWidth = img.clientWidth;
        const imageHeight = img.clientHeight;

        state.originalImageDimensions = { width: imageWidth, height: imageHeight };
        state.viewportCenterX = imageWidth / 2;
        state.viewportCenterY = imageHeight / 2;

        // Resize all 3 canvases
        [annotationCanvas, predictionCanvas, medsamCanvas].forEach(c => {
            c.width = imageWidth;
            c.height = imageHeight;
            c.style.width = `${imageWidth}px`;
            c.style.height = `${imageHeight}px`;
            c.style.position = "absolute";
            c.style.top = "0";
            c.style.left = "0";
        });

        // Zoom container
        const zoomContainer = document.getElementById("zoomContainer");
        if (zoomContainer) {
            zoomContainer.style.width = `${imageWidth}px`;
            zoomContainer.style.height = `${imageHeight}px`;
        }

        redrawAnnotations(); // Repaint with correct scaling
    };

    // Ensure resize happens when image loads
    if (img.complete) {
        resizeCanvasToImage();
    } else {
        img.onload = resizeCanvasToImage;
    }

    // Optional: respond to browser resizes
    window.addEventListener("resize", resizeCanvasToImage);

    // Hook event handlers
    window.handleAnnotations = handleAnnotations;
    window.handlePreprocessedImg = handlePreprocessedImg;
    window.loadAnnotations = loadAnnotations;
    window.downloadAnnotations = downloadAnnotations;

    bindUIEvents();
    initColorPicker();
    initBoxTool();

    // Adjust UI for selected method
    const segmentBtn = document.querySelector('.ready-segment-btn');
    if (segmentBtn) {
        if (state.unetMode) {
            segmentBtn.innerHTML = '<i class="fa-solid fa-hurricane me-2"></i> Ready to Segment with UNet!';
        } else if (state.medsamMode) {
            segmentBtn.innerHTML = '<i class="fa-solid fa-hurricane me-2"></i> Ready to Segment with MedSAM!';
        } else {
            segmentBtn.innerHTML = '<i class="fa-solid fa-hurricane me-2"></i> Ready to Segment!';
        }
    }

    if (typeof updateMethodDescription === 'function') {
        updateMethodDescription();
    }
});
