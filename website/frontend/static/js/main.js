// main.js
import { state, initializeFromServer } from './state.js';
import { bindUIEvents, updateMethodDescription } from './events.js';
import { redrawAnnotations } from './canvas-tools.js';
import { initColorPicker } from './color-picker.js';
import { handleAnnotations, handlePreprocessedImg } from './api-service.js';
import { initializeAnnotationsFromPredictions } from './canvas-utils.js';


document.addEventListener('DOMContentLoaded', () => {
    const annotationCanvas = document.getElementById("annotationCanvas");
    const predictionCanvas = document.getElementById("predictionCanvas");
    const img = document.getElementById("uploadedImage");

    state.annotationCanvas = annotationCanvas;
    state.predictionCanvas = predictionCanvas;
    state.annotationCtx = annotationCanvas.getContext("2d");
    state.predictionCtx = predictionCanvas.getContext("2d");
    state.imageName = window.imageName;

    if (window.segmentationMethod) {
        state.segmentationMethod = window.segmentationMethod;
        state.unetMode = window.segmentationMethod === 'unet';
    }

    if (typeof initializeFromServer === 'function') {
        initializeFromServer({
            imageName: window.imageName,
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

    updateMethodDescription();

    if (state.unetMode) {
        const segmentBtn = document.querySelector('.ready-segment-btn');
        if (segmentBtn) {
            segmentBtn.innerHTML = '<i class="fa-solid fa-hurricane me-2"></i> Ready to Segment with UNet!';
        }
    }
});

//Debug
window.testFunction = function() {
    alert('JavaScript is working!');
};

