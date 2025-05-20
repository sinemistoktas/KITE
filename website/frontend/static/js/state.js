export const state = {
    // Interaction state
    mouseX: 0,
    mouseY: 0,
    mode: null,
    showEraserCursor: false,
    isDrawing: false,
    isErasing: false,
    isFillToolActive: false,
    isDrawingBoundary: false,
    boundaryComplete: false,

    // Drawing and annotation
    currentStroke: [],
    selectedColor: "#ff0000",
    currentStrokeColor: "#ff0000",
    currentBoundary: [],
    scribbles: [],
    layerCounter: 0,
    currentLayerId: null,
    visibleLayerIds: [],
    showPredictions: true,

    // Zooming
    zoomLevel: 1,
    zoomMode: false,
    viewportCenterX: 0,
    viewportCenterY: 0,
    originalImageDimensions: { width: 0, height: 0 },

    // Canvas references (will be initialized later)
    annotationCanvas: null,
    annotationCtx: null,
    predictionCanvas: null,
    predictionCtx: null,

    //UNet integration
    unetMode: false,
    imageName: null,
    predictedPoints: [],
    segmentationMethod: null,
};

export const ERASE_RADIUS = 10;

export function isUNetMode() {
    return state.unetMode;
}

export function initializeFromServer(serverData) {
    if (serverData.imageName) {
        state.imageName = serverData.imageName;
    }
    if (serverData.segmentationMethod) {
        state.segmentationMethod = serverData.segmentationMethod;
        state.unetMode = serverData.segmentationMethod === 'unet';
    }
    if (serverData.predictedPoints) {
        state.predictedPoints = serverData.predictedPoints;
    }
}