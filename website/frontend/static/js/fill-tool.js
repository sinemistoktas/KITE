import { state } from './state.js';
import { screenToImageCoords } from './canvas-utils.js';
import { redrawAnnotations } from './canvas-tools.js';
import { createLayer } from './layers.js';

export function toggleFillTool() {
    if (state.mode !== null) {
        state.mode = null;
    }

    if (state.zoomMode) {
        state.zoomMode = false;
        document.getElementById('zoomInBtn')?.classList.remove('btn-primary');
        document.getElementById('zoomInBtn')?.classList.add('btn-outline-primary');
        state.annotationCanvas.classList.remove('zoom-cursor');
        state.annotationCanvas.style.cursor = 'crosshair';
    }

    state.isFillToolActive = !state.isFillToolActive;

    const fillBtn = document.getElementById('fillToolBtn');
    const status = document.getElementById('fillToolStatus');

    if (state.isFillToolActive) {
        fillBtn?.classList.add('btn-danger');
        fillBtn?.classList.remove('btn-outline-danger');
        resetFillTool();
        updateFillToolStatus("Draw a closed boundary around the area you want to fill");
    } else {
        fillBtn?.classList.remove('btn-danger');
        fillBtn?.classList.add('btn-outline-danger');
        if (status) status.style.display = 'none';
    }

    redrawAnnotations();
}

export function handleFillToolClick(imageCoords) {
    if (!state.boundaryComplete) {
        state.isDrawingBoundary = true;
        state.currentBoundary = [{ x: imageCoords.x, y: imageCoords.y }];
    } else {
        if (!isPointInPolygon(imageCoords, state.currentBoundary)) {
            updateFillToolStatus("Click must be inside the boundary. Try again.");
            return;
        }

        const layerId = createLayer("Fill", state.selectedColor);
        state.scribbles.push({
            points: [...state.currentBoundary],
            isPrediction: false,
            color: state.selectedColor,
            layerId
        });

        updateFillToolStatus("Filling region with dots... please wait");

        setTimeout(() => {
            const fillPoints = floodFill(imageCoords);
            fillPoints.forEach(point => {
                state.scribbles.push({
                    points: [point],
                    isPrediction: false,
                    color: state.selectedColor,
                    layerId
                });
            });

            resetFillTool();
            updateFillToolStatus("Fill complete. Draw another boundary or switch tools.");
            redrawAnnotations();
        }, 50);
    }
}

function floodFill(centerPoint) {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = state.annotationCanvas.width;
    tempCanvas.height = state.annotationCanvas.height;
    const ctx = tempCanvas.getContext('2d');

    const offsetX = state.viewportCenterX - (state.originalImageDimensions.width / (2 * state.zoomLevel));
    const offsetY = state.viewportCenterY - (state.originalImageDimensions.height / (2 * state.zoomLevel));

    ctx.save();
    ctx.translate(-offsetX * state.zoomLevel, -offsetY * state.zoomLevel);
    ctx.scale(state.zoomLevel, state.zoomLevel);
    ctx.beginPath();
    ctx.moveTo(state.currentBoundary[0].x, state.currentBoundary[0].y);
    for (let i = 1; i < state.currentBoundary.length; i++) {
        ctx.lineTo(state.currentBoundary[i].x, state.currentBoundary[i].y);
    }
    ctx.closePath();
    ctx.fill();
    ctx.restore();

    const imageData = ctx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
    const data = imageData.data;
    const sampleRate = 5;
    const filledPoints = [];

    for (let y = 0; y < tempCanvas.height; y += sampleRate) {
        for (let x = 0; x < tempCanvas.width; x += sampleRate) {
            const index = (y * tempCanvas.width + x) * 4;
            if (data[index + 3] > 0) {
                filledPoints.push(screenToImageCoords(x + state.annotationCanvas.getBoundingClientRect().left, y + state.annotationCanvas.getBoundingClientRect().top));
            }
        }
    }

    return filledPoints;
}

export function updateFillToolStatus(msg) {
    const status = document.getElementById('fillToolStatus');
    if (status) {
        status.textContent = msg;
        status.style.display = 'block';
    }
}

export function resetFillTool() {
    state.isDrawingBoundary = false;
    state.currentBoundary = [];
    state.boundaryComplete = false;
}

function isPointInPolygon(p, poly) {
    let inside = false;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
        const xi = poly[i].x, yi = poly[i].y;
        const xj = poly[j].x, yj = poly[j].y;

        const intersect = ((yi > p.y) !== (yj > p.y)) &&
            (p.x < (xj - xi) * (p.y - yi) / (yj - yi) + xi);
        if (intersect) inside = !inside;
    }
    return inside;
}