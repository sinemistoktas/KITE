// website/frontend/static/js/fill-tool.js

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
        const fillStroke = {
            points: [...state.currentBoundary],
            isPrediction: false,
            color: state.selectedColor,
            layerId,
            isFilled: true,
            type: 'fill'
        };
        state.scribbles.push(fillStroke);

        resetFillTool();
        updateFillToolStatus("Fill complete. Draw another boundary or switch tools.");
        redrawAnnotations();
    }
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