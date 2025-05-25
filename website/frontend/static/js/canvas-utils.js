import { state, ERASE_RADIUS } from './state.js';

let createLayer;
let redrawAnnotations;

export function setDependencies(deps) {
    if (deps.createLayer) createLayer = deps.createLayer;
    if (deps.redrawAnnotations) redrawAnnotations = deps.redrawAnnotations;
}

export function distance(p, x, y) {
    const dx = p.x - x;
    const dy = p.y - y;
    return (dx * dx + dy * dy) <= (ERASE_RADIUS * ERASE_RADIUS);
}

export function screenToImageCoords(screenX, screenY) {
    const rect = state.annotationCanvas.getBoundingClientRect();
    const containerX = screenX - rect.left;
    const containerY = screenY - rect.top;

    const offsetX = state.viewportCenterX - (state.originalImageDimensions.width / (2 * state.zoomLevel));
    const offsetY = state.viewportCenterY - (state.originalImageDimensions.height / (2 * state.zoomLevel));

    const imageX = (containerX / state.zoomLevel) + offsetX;
    const imageY = (containerY / state.zoomLevel) + offsetY;

    return {
        x: Math.max(0, Math.min(state.originalImageDimensions.width, imageX)),
        y: Math.max(0, Math.min(state.originalImageDimensions.height, imageY))
    };
}



