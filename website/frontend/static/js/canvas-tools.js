import { state } from './state.js';
import { distance } from './canvas-utils.js';
import { drawBoxPreview } from './box-tool.js';

export function redrawAnnotations() {
    const aCtx = state.annotationCtx;
    const pCtx = state.predictionCtx;

    aCtx.clearRect(0, 0, state.annotationCanvas.width, state.annotationCanvas.height);
    pCtx.clearRect(0, 0, state.predictionCanvas.width, state.predictionCanvas.height);

    const adjustForZoom = (ctx) => {
        ctx.save();
        const offsetX = state.viewportCenterX - (state.originalImageDimensions.width / (2 * state.zoomLevel));
        const offsetY = state.viewportCenterY - (state.originalImageDimensions.height / (2 * state.zoomLevel));
        ctx.translate(-offsetX * state.zoomLevel, -offsetY * state.zoomLevel);
        ctx.scale(state.zoomLevel, state.zoomLevel);
    };

    const restoreZoom = (ctx) => ctx.restore();

    // Draw user annotations
    state.scribbles
        .filter(s => !s.isPrediction && (!s.layerId || state.visibleLayerIds.includes(s.layerId)))
        .forEach(stroke => {
            adjustForZoom(aCtx);

            if (stroke.isLoadedAnnotation || stroke.isDot || stroke.isSegmentationResult || stroke.points.length === 1) {
                aCtx.fillStyle = stroke.color || "red";

                const radius = (stroke.radius || 2) / state.zoomLevel;
                stroke.points.forEach(point => {
                    aCtx.beginPath();
                    aCtx.arc(point.x, point.y, radius, 0, 2 * Math.PI);
                    aCtx.fill();
                });
            }
            else if (stroke.isFilled || stroke.type === 'fill') {
                aCtx.fillStyle = stroke.color || "red";
                aCtx.beginPath();
                aCtx.moveTo(stroke.points[0].x, stroke.points[0].y);
                for (let i = 1; i < stroke.points.length; i++) {
                    aCtx.lineTo(stroke.points[i].x, stroke.points[i].y);
                }
                aCtx.closePath();
                aCtx.fill();

                aCtx.strokeStyle = stroke.color || "red";
                aCtx.lineWidth = 1 / state.zoomLevel;
                aCtx.stroke();
            }
            else if (stroke.isBox) {
                aCtx.strokeStyle = stroke.color || "red";
                aCtx.lineWidth = 2 / state.zoomLevel;
                aCtx.beginPath();
                aCtx.moveTo(stroke.points[0].x, stroke.points[0].y);
                for (let i = 1; i < stroke.points.length; i++) {
                    aCtx.lineTo(stroke.points[i].x, stroke.points[i].y);
                }
                aCtx.stroke();
            }
            else {
                aCtx.strokeStyle = stroke.color || "red";
                aCtx.lineWidth = (stroke.thickness || 2) / state.zoomLevel;
                aCtx.beginPath();
                aCtx.moveTo(stroke.points[0].x, stroke.points[0].y);
                for (let i = 1; i < stroke.points.length; i++) {
                    aCtx.lineTo(stroke.points[i].x, stroke.points[i].y);
                }
                aCtx.stroke();
            }

            restoreZoom(aCtx);
        });

    // Draw predictions
    if (state.showPredictions && !state.isEditingSegmentationResults) {
        const predictionStrokes = state.scribbles.filter(s => s.isPrediction);
        if (predictionStrokes.length > 0) {
            adjustForZoom(pCtx);
            pCtx.strokeStyle = "blue";
            pCtx.lineWidth = 2 / state.zoomLevel;

            for (const stroke of predictionStrokes) {
                for (const point of stroke.points) {
                    pCtx.fillStyle = point.color || stroke.color || "blue";
                    pCtx.beginPath();
                    pCtx.arc(point.x, point.y, 2 / state.zoomLevel, 0, 2 * Math.PI);
                    pCtx.fill();
                }
            }

            restoreZoom(pCtx);
        }
    }

    // Draw current stroke
    if (state.isDrawing && state.currentStroke.length > 0) {
        adjustForZoom(aCtx);
        aCtx.strokeStyle = state.currentStrokeColor;
        aCtx.lineWidth = (state.lineThickness || 2) / state.zoomLevel;
        aCtx.beginPath();
        aCtx.moveTo(state.currentStroke[0].x, state.currentStroke[0].y);
        for (let i = 1; i < state.currentStroke.length; i++) {
            aCtx.lineTo(state.currentStroke[i].x, state.currentStroke[i].y);
        }
        aCtx.stroke();
        restoreZoom(aCtx);
    }

    // Draw box preview
    drawBoxPreview(aCtx, adjustForZoom, restoreZoom);

    // Draw eraser cursor
    if (state.showEraserCursor && state.mode === "eraser") {
        adjustForZoom(aCtx);
        aCtx.strokeStyle = "rgba(0,255,255,0.9)";
        aCtx.lineWidth = 1 / state.zoomLevel;
        const radius = (state.eraserRadius || 10);
        aCtx.beginPath();
        aCtx.arc(state.mouseX, state.mouseY, radius, 0, 2 * Math.PI);
        aCtx.stroke();
        restoreZoom(aCtx);
    }

    // Draw fill boundary
    if (state.isFillToolActive && state.currentBoundary.length > 1) {
        adjustForZoom(aCtx);
        aCtx.strokeStyle = state.selectedColor;
        aCtx.lineWidth = 2 / state.zoomLevel;
        aCtx.beginPath();
        aCtx.moveTo(state.currentBoundary[0].x, state.currentBoundary[0].y);
        for (let i = 1; i < state.currentBoundary.length; i++) {
            aCtx.lineTo(state.currentBoundary[i].x, state.currentBoundary[i].y);
        }
        if (state.boundaryComplete) aCtx.closePath();
        aCtx.stroke();
        restoreZoom(aCtx);
    }

    if (state.isEditingSegmentationResults) {
        const segmentationStrokes = state.scribbles.filter(s => s.isSegmentationResult);
        console.log(`🏏 Editing mode: Rendering ${segmentationStrokes.length} editable segmentation strokes, predictions hidden`);
    }
}

export function zoomToPoint(x, y) {
    if (state.zoomLevel >= 5) return;
    const newZoom = Math.min(5, state.zoomLevel * 1.2);
    const halfWidth = state.originalImageDimensions.width / (2 * newZoom);
    const halfHeight = state.originalImageDimensions.height / (2 * newZoom);

    state.viewportCenterX = Math.max(halfWidth, Math.min(state.originalImageDimensions.width - halfWidth, x));
    state.viewportCenterY = Math.max(halfHeight, Math.min(state.originalImageDimensions.height - halfHeight, y));
    state.zoomLevel = newZoom;

    updateZoom();
}

export function zoomOut() {
    if (state.zoomLevel <= 1) return;
    const newZoom = Math.max(1, state.zoomLevel / 1.2);

    const halfWidth = state.originalImageDimensions.width / (2 * newZoom);
    const halfHeight = state.originalImageDimensions.height / (2 * newZoom);

    state.viewportCenterX = Math.max(halfWidth, Math.min(state.originalImageDimensions.width - halfWidth, state.viewportCenterX));
    state.viewportCenterY = Math.max(halfHeight, Math.min(state.originalImageDimensions.height - halfHeight, state.viewportCenterY));
    state.zoomLevel = newZoom;

    updateZoom();
}

export function resetZoom() {
    state.zoomLevel = 1;
    state.viewportCenterX = state.originalImageDimensions.width / 2;
    state.viewportCenterY = state.originalImageDimensions.height / 2;
    updateZoom();
}

export function updateZoom() {
    const img = document.getElementById("uploadedImage");
    const offsetX = state.viewportCenterX - (state.originalImageDimensions.width / (2 * state.zoomLevel));
    const offsetY = state.viewportCenterY - (state.originalImageDimensions.height / (2 * state.zoomLevel));

    img.style.transform = `scale(${state.zoomLevel})`;
    img.style.transformOrigin = "top left";
    img.style.position = "absolute";
    img.style.left = `${-offsetX * state.zoomLevel}px`;
    img.style.top = `${-offsetY * state.zoomLevel}px`;

    state.annotationCanvas.style.transform = "";
    state.predictionCanvas.style.transform = "";

    redrawAnnotations();

    const zoomInfo = document.getElementById("zoomInfo");
    if (zoomInfo) {
        zoomInfo.textContent = `Zoom: ${Math.round(state.zoomLevel * 100)}%`;
    }
}