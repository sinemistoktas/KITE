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

export function initializeAnnotationsFromPredictions(predictions) {
    if (!predictions || !predictions.length) return;

    import('./layers.js').then(module => {
        const { createLayer } = module;

        const layerId = createLayer("UNet", "#ff0000");

        predictions.forEach(shape => {
            if (shape.points && shape.points.length > 0) {
                const points = shape.points.map(point => ({
                    x: point[0],
                    y: point[1]
                }));

                state.scribbles.push({
                    points: points,
                    isPrediction: false, // Treat as a normal annotation that can be edited
                    color: shape.color || "#ff0000",
                    layerId: layerId
                });
            }
        });

        import('./canvas-tools.js').then(module => {
            const { redrawAnnotations } = module;
            redrawAnnotations();
        });
    });
}

export function processUNetPredictions(predictedPoints) {
    if (!predictedPoints || predictedPoints.length === 0) return [];

    return predictedPoints.map(point => {
        if (Array.isArray(point[0])) {
            const [[x, y], color] = point;
            return { x, y, color };
        } else {
            const [x, y] = point;
            return { x, y, color: "#ff0000" };
        }
    });
}