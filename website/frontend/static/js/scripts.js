/*!
* Start Bootstrap - Resume v7.0.6 (https://startbootstrap.com/theme/resume)
* Copyright 2013-2023 Start Bootstrap
* Licensed under MIT (https://github.com/StartBootstrap/startbootstrap-resume/blob/master/LICENSE)
*/
//
// Scripts
// 

let mouseX = 0;
let mouseY = 0;
let showEraserCursor = false;
let mode = null; // The current mode of the tool: Line, dot, or eraser.
let isFillToolActive = false;  // Track if we're in fill tool mode
let isDrawingBoundary = false; // Track if drawing the boundary
let currentBoundary = [];      // Store the current boundary points
let boundaryComplete = false;  // Track if a boundary is complete
let isDrawing = false;
let isErasing = false;
const ERASE_RADIUS = 10; // Could be changed to make it smaller?
let annotationCanvas, annotationCtx, predictionCanvas, predictionCtx; // There are two different canvases: One for 
// the algorithm's predictions (immutable), and one for the user's annotations.

// Global variables for zooming
let zoomLevel = 1;
let zoomMode = false;
let viewportCenterX = 0; // The CENTER of the current view. Will be useful for zooming.
let viewportCenterY = 0;
let originalImageDimensions = { width: 0, height: 0 };

// GLOBAL references to be set on DOMContentLoaded
let scribbles = [], currentStroke = [];
let selectedColor = "#ff0000"; // def annotation color is red
let currentStrokeColor = "#ff0000";
let layerCounter = 0; // Counter for unique layer IDs
let currentLayerId;
let visibleLayerIds = []; // Add this at the top with other global variables
let showPredictions = true; // Add this with other global variables

// Useful for the eraser feature.
function distance(p, x, y) {
    const dx = p.x - x;
    const dy = p.y - y;
    return (dx * dx + dy * dy) <= (ERASE_RADIUS * ERASE_RADIUS);
}


// Converts the mouse clicks on the screen into coordinates directly on the image.
function screenToImageCoords(screenX, screenY) {
    const rect = annotationCanvas.getBoundingClientRect();
    // Ensures that the current coordinates are relative to the image and not the head div.
    const containerX = screenX - rect.left;
    const containerY = screenY - rect.top;
    
    // Adds the zoom effect.
    const offsetX = viewportCenterX - (originalImageDimensions.width / (2 * zoomLevel));
    const offsetY = viewportCenterY - (originalImageDimensions.height / (2 * zoomLevel));
    
    // Calculate the exact location on the image.
    const imageX = (containerX / zoomLevel) + offsetX;
    const imageY = (containerY / zoomLevel) + offsetY;
    
    // Here, we have to apply constraints to make sure that invalid values (negative values) do not occur.
    const constrainedX = Math.max(0, Math.min(originalImageDimensions.width, imageX));
    const constrainedY = Math.max(0, Math.min(originalImageDimensions.height, imageY));
    
    return { x: constrainedX, y: constrainedY };
}

window.addEventListener('DOMContentLoaded', event => {

    const fillToolBtn = document.getElementById('fillToolBtn');
    
    // Add event listener for the fill tool button
    fillToolBtn?.addEventListener('click', toggleFillTool);
    
    // Add keyboard shortcut for fill tool (F)
    document.addEventListener('keydown', function(e) {
        if (["INPUT", "TEXTAREA"].includes(document.activeElement.tagName)) {
            return;
        }
        
        if (e.key.toUpperCase() === 'F') {
            toggleFillTool();
        }
    });

    annotationCanvas = document.getElementById("annotationCanvas");
    annotationCtx = annotationCanvas.getContext("2d");

    predictionCanvas = document.getElementById("predictionCanvas");
    predictionCtx = predictionCanvas.getContext("2d");

    // Color picker functionality
    const colorPicker = document.getElementById("colorPicker");
    if (colorPicker) {
        // Set initial color
        selectedColor = colorPicker.value;
        
        // Make sure the color picker is properly configured
        colorPicker.style.cursor = 'pointer';
        colorPicker.style.width = '38px';
        colorPicker.style.height = '38px';
        colorPicker.style.padding = '0';
        colorPicker.style.border = 'none';
        colorPicker.style.borderRadius = '8px';
        
        // Handle color changes from the toolbox color picker
        colorPicker.addEventListener("input", function(e) {
            selectedColor = e.target.value;
        });
    }

    // Draw line function from first script
    function drawLine(points, color = selectedColor, context = annotationCtx, lineWidth = 2) {
        if (points.length < 2) return;
        context.strokeStyle = color;
        context.lineWidth = lineWidth;
        context.beginPath();
        context.moveTo(points[0].x, points[0].y);
        for (let i = 1; i < points.length; i++) {
            context.lineTo(points[i].x, points[i].y);
        }
        context.stroke();
    }

    function eraseAtPoint(x, y) {
        let newScribbles = [];
        let existingLayerIds = new Set(); // Track all layer IDs that exist, to be used for deleting layers that no longer have any strokes.
        let erasedStroke = false;

        // First pass: collect all layer IDs
        for (const stroke of scribbles) {
            if (stroke.layerId) {
                existingLayerIds.add(stroke.layerId);
            }
        }

        // Process erasing
        for (const stroke of scribbles) {
            if (stroke.isPrediction) {
                newScribbles.push(stroke); // keep the made predictions
                continue;
            }

            // Check if stroke has any points within eraser radius
            const hasPointInEraser = stroke.points.some(point => distance(point, x, y));

            if (!hasPointInEraser || erasedStroke) {
                newScribbles.push(stroke);
                continue;
            }

            erasedStroke = true;

            let currentSegment = [];
            let hasRemainingPoints = false;

            for (const point of stroke.points) {
                if (!distance(point,x,y)) {
                    currentSegment.push(point);
                    hasRemainingPoints = true;
                }
                else {
                    // if eraser hits a point, break the stroke here
                    if (currentSegment.length > 1) {
                        newScribbles.push({
                            points: currentSegment,
                            isPrediction: false,
                            color: stroke.color,
                            layerId: stroke.layerId
                        });
                    }
                    currentSegment = [];
                }
            }

            if (currentSegment.length > 1) {
                newScribbles.push({
                    points: currentSegment,
                    isPrediction: false,
                    color: stroke.color,
                    layerId: stroke.layerId
                });
            }
        }

        scribbles = newScribbles;

        // Check which layers still have strokes
        let remainingLayerIds = new Set();
        for (const stroke of newScribbles) {
            if (stroke.layerId) {
                remainingLayerIds.add(stroke.layerId);
            }
        }

        // Delete layers that no longer have any strokes
        for (const layerId of existingLayerIds) {
            if (!remainingLayerIds.has(layerId)) {
                const layerElement = document.getElementById(layerId);
                if (layerElement) {
                    layerElement.remove();
                }
            }
        }

        redrawAnnotations(); // Redraw everything
    }
    

    function drawEraserCursor() {
        if (!showEraserCursor || mode !== "eraser") return;
        redrawAnnotations();
    }

    const img = document.getElementById("uploadedImage");

    // Activate Bootstrap scrollspy on the main nav element
    const sideNav = document.body.querySelector('#sideNav');
    if (sideNav) {
        new bootstrap.ScrollSpy(document.body, {
            target: '#sideNav',
            rootMargin: '0px 0px -40%',
        });
    };

    // Collapse responsive navbar when toggler is visible
    const navbarToggler = document.body.querySelector('.navbar-toggler');
    const responsiveNavItems = [].slice.call(
        document.querySelectorAll('#navbarResponsive .nav-link')
    );
    responsiveNavItems.map(function (responsiveNavItem) {
        responsiveNavItem.addEventListener('click', () => {
            if (window.getComputedStyle(navbarToggler).display !== 'none') {
                navbarToggler.click();
            }
        });
    });

    // Sets the style of the canvases to ensure that it is completely aligned with the image.
    function resizeCanvasToImage() {
        const rect = img.getBoundingClientRect();
        originalImageDimensions = { width: rect.width, height: rect.height };
        
        // Set initial zooming center to the center of the image.
        viewportCenterX = originalImageDimensions.width / 2;
        viewportCenterY = originalImageDimensions.height / 2;
        
        // Initialize the canvases to have the same exact size as the image's proportions.
        if (zoomLevel === 1) {
            for (const c of [annotationCanvas, predictionCanvas]) {
                c.width = rect.width;
                c.height = rect.height;
                c.style.width = `${rect.width}px`;
                c.style.height = `${rect.height}px`;
                c.style.position = "absolute";
                c.style.top = "0";
                c.style.left = "0";
                c.style.pointerEvents = "none";
            }
            
            annotationCanvas.style.pointerEvents = "auto"; // Allows ONLY the annotation layer to be interactive.
            
            // Initialize zoom container dimensions, should again be the same as the image.
            const zoomContainer = document.getElementById("zoomContainer");
            if (zoomContainer) {
                zoomContainer.style.width = `${rect.width}px`;
                zoomContainer.style.height = `${rect.height}px`;
            }
        }
    }
    //TODO: When the window is resized the canvas is DELETED. what to do ?
      
    if (img.complete) { // If the browser has finished loading the image, resize the canvas. Else resize it when it loads.
        resizeCanvasToImage();
    } else {
        img.onload = () => {
            resizeCanvasToImage();
        };
    }

    window.addEventListener("resize", resizeCanvasToImage); // Whenever the browser window & the image is resized, the canvas would resize with it.


    const lineBtn = document.getElementById("scribbleMode");
    const dotBtn = document.getElementById("dotMode");
    const eraserBtn = document.getElementById("eraserMode");
    const eraseAllBtn = document.getElementById("eraseAllMode");
    const zoomInBtn = document.getElementById("zoomInBtn");
    const zoomOutBtn = document.getElementById("zoomOutBtn");
    const resetZoomBtn = document.getElementById("resetZoomBtn");

    // Function to change the selected button's color to red.
    function updateButtonStyles() {
        if (mode === "line") {
            lineBtn.classList.remove("btn-outline-danger");
            lineBtn.classList.add("btn-danger");
            dotBtn.classList.remove("btn-danger");
            dotBtn.classList.add("btn-outline-danger");
            eraserBtn.classList.remove("btn-danger");
            eraserBtn.classList.add("btn-outline-danger");
            eraseAllBtn.classList.remove("btn-danger");
            eraseAllBtn.classList.add("btn-outline-danger");
        } else if (mode === "dot") {
            dotBtn.classList.remove("btn-outline-danger");
            dotBtn.classList.add("btn-danger");
            lineBtn.classList.remove("btn-danger");
            lineBtn.classList.add("btn-outline-danger");
            eraserBtn.classList.remove("btn-danger");
            eraserBtn.classList.add("btn-outline-danger");
            eraseAllBtn.classList.remove("btn-danger");
            eraseAllBtn.classList.add("btn-outline-danger");
        } else if (mode === "eraser") {
            eraserBtn.classList.remove("btn-outline-danger");
            eraserBtn.classList.add("btn-danger");
            lineBtn.classList.remove("btn-danger");
            lineBtn.classList.add("btn-outline-danger");
            dotBtn.classList.remove("btn-danger");
            dotBtn.classList.add("btn-outline-danger");
            eraseAllBtn.classList.remove("btn-danger");
            eraseAllBtn.classList.add("btn-outline-danger");
        } else if (mode === "eraseAll") {
            eraseAllBtn.classList.remove("btn-outline-danger");
            eraseAllBtn.classList.add("btn-danger");
            lineBtn.classList.remove("btn-danger");
            lineBtn.classList.add("btn-outline-danger");
            dotBtn.classList.remove("btn-danger");
            dotBtn.classList.add("btn-outline-danger");
            eraserBtn.classList.remove("btn-danger");
            eraserBtn.classList.add("btn-outline-danger");
        }
        else { // Deselect all buttons if the user has clicked on a button that they have already clicked on.
            lineBtn.classList.remove("btn-danger");
            lineBtn.classList.add("btn-outline-danger");
            dotBtn.classList.remove("btn-danger");
            dotBtn.classList.add("btn-outline-danger");
            eraserBtn.classList.remove("btn-danger");
            eraserBtn.classList.add("btn-outline-danger");
            eraseAllBtn.classList.remove("btn-danger");
            eraseAllBtn.classList.add("btn-outline-danger");
        }
        
        // Same idea, but for the zoom buttons.
        if (zoomMode) {
            zoomInBtn.classList.remove("btn-outline-primary");
            zoomInBtn.classList.add("btn-primary");
            annotationCanvas.classList.add("zoom-cursor");
        } else {
            zoomInBtn.classList.remove("btn-primary");
            zoomInBtn.classList.add("btn-outline-primary");
            annotationCanvas.classList.remove("zoom-cursor");
        }
    }

    // A more generic function to set the mode rather than the previous calls. This ensures that a button does not need to be
    // clicked again to reset its mode's properties, like the moving eraser cursor.

    function setMode(newMode) {
        zoomInBtn.classList.remove("active");
        zoomOutBtn.classList.remove("active");
        resetZoomBtn.classList.remove("active");
    
        // If fill tool is active, deactivate it when switching to another mode
        if (isFillToolActive && newMode !== null) {
            isFillToolActive = false;
            const fillBtn = document.getElementById('fillToolBtn');
            if (fillBtn) {
                fillBtn.classList.remove('btn-danger');
                fillBtn.classList.add('btn-outline-danger');
            }
            const fillStatus = document.getElementById('fillToolStatus');
            if (fillStatus) {
                fillStatus.style.display = 'none';
            }
            resetFillTool();
        }
    
        if (mode === newMode) {
            // Deselect if user clicks the same button again.
            mode = null;
            showEraserCursor = false;
            if (newMode === "eraseAll") {
                eraseAllAnnotations();
            }
        } else {
            // Exit zoom mode if we're switching to a drawing mode
            if (zoomMode) {
                zoomMode = false;
                updateButtonStyles();
            }
            
            const previousMode = mode;
            mode = newMode;
            showEraserCursor = (newMode === "eraser");
    
            // If switching FROM eraser mode, clear the eraser cursor from canvas.
            if (previousMode === "eraser" && newMode !== "eraser") {
                annotationCanvas.style.cursor = "crosshair";
                redrawAnnotations();
            }
            
            // Update cursor visibility based on mode.
            if (newMode === "eraser") {
                annotationCanvas.style.cursor = "none";
            } else if (newMode === "eraseAll") {
                eraseAllAnnotations();
            } else {
                annotationCanvas.style.cursor = "crosshair";
            }
        }
    
        updateButtonStyles();
    }
    
    // Function to toggle the zoom mode.
    function toggleZoomMode() {
        zoomMode = !zoomMode;
        
        if (zoomMode) {
            // Deactivate any other mode, including fill tool
            mode = null;
            showEraserCursor = false;
            
            // Deselect fill tool if it's active
            if (isFillToolActive) {
                isFillToolActive = false;
                const fillBtn = document.getElementById('fillToolBtn');
                if (fillBtn) {
                    fillBtn.classList.remove('btn-danger');
                    fillBtn.classList.add('btn-outline-danger');
                }
                const fillStatus = document.getElementById('fillToolStatus');
                if (fillStatus) {
                    fillStatus.style.display = 'none';
                }
                resetFillTool();
            }
            
            annotationCanvas.style.cursor = "zoom-in";
        } else {
            annotationCanvas.style.cursor = "crosshair";
        }
        
        updateButtonStyles();
    }
    
    
    // Zoom in function, the function argument is the x and y coordinates of what the user has clicked on
    // to zoom.
    function zoomToPoint(x, y) {
        if (zoomLevel >= 5) return; // This is the maximum zoom level, which is currently 500%.
        
        // Again, we can change this. Currently, at each zoom, the zoom level is factored by 1.2.
        const newZoomLevel = Math.min(5, zoomLevel * 1.2);
        
        // The sizes here are calculated to make sure that the zoom doesn't go out of bounds, and show
        // white parts for example.
        const horizontalViewSize = originalImageDimensions.width / (2 * newZoomLevel);
        const verticalViewSize = originalImageDimensions.height / (2 * newZoomLevel);
        
        // These are the minimum and maximum x and y points allowed as the center points of the zoomed in regions.
        const minX = horizontalViewSize;
        const maxX = originalImageDimensions.width - horizontalViewSize;
        const minY = verticalViewSize;
        const maxY = originalImageDimensions.height - verticalViewSize;
        
        // Find the new zoom center.
        viewportCenterX = Math.max(minX, Math.min(maxX, x));
        viewportCenterY = Math.max(minY, Math.min(maxY, y));
        
        zoomLevel = newZoomLevel;
        
        updateZoom();
    }
    
    function zoomOut() {
        if (zoomLevel <= 1) return; // I set the minimum zoom level to 100% as I don't think doctors would want to zoom
        // out of an OCT image (maybe?..)
        
        // Same as zooming in, zooming out divides by 1.2.
        const newZoomLevel = Math.max(1, zoomLevel / 1.2);
        
        // If we're going back to zoom level 1, reset to center.
        if (Math.abs(newZoomLevel - 1) < 0.05) {
            zoomLevel = 1;
            viewportCenterX = originalImageDimensions.width / 2;
            viewportCenterY = originalImageDimensions.height / 2;
        } else {
            // Same as zooming in, apply these constraints to make sure that when we zoom out, we don't 
            // cross boundaries.
            const horizontalViewSize = originalImageDimensions.width / (2 * newZoomLevel);
            const verticalViewSize = originalImageDimensions.height / (2 * newZoomLevel);
            
            const minX = horizontalViewSize;
            const maxX = originalImageDimensions.width - horizontalViewSize;
            const minY = verticalViewSize;
            const maxY = originalImageDimensions.height - verticalViewSize;
            
            viewportCenterX = Math.max(minX, Math.min(maxX, viewportCenterX));
            viewportCenterY = Math.max(minY, Math.min(maxY, viewportCenterY));
            
            // Apply the new zoom level.
            zoomLevel = newZoomLevel;
        }
        
        updateZoom();
    }
    
    // Will be called when we click on "reset zoom", basically resets the image back to the zoom level of 100%.
    function resetZoom() {
        zoomLevel = 1;
        viewportCenterX = originalImageDimensions.width / 2;
        viewportCenterY = originalImageDimensions.height / 2;
        updateZoom();
    }
    
    // Apply the new zoom level.
    function updateZoom() {
        const zoomContainer = document.getElementById("zoomContainer");
        
        // Calculate the offset needed to center the view on the zoomed point.
        const offsetX = viewportCenterX - (originalImageDimensions.width / (2 * zoomLevel));
        const offsetY = viewportCenterY - (originalImageDimensions.height / (2 * zoomLevel));
        
        // Transforms the size of the image, we MUST keep the canvas fixed.
        img.style.transform = `scale(${zoomLevel})`;
        img.style.transformOrigin = "top left";
        img.style.position = "absolute";
        img.style.left = `${-offsetX * zoomLevel}px`;
        img.style.top = `${-offsetY * zoomLevel}px`;
        
        // NOTHING should happen to the canvases.
        predictionCanvas.style.transform = "";
        annotationCanvas.style.transform = "";
        
        // Finally, we rerender the annotations.
        redrawAnnotations();
        
        // Update the zoom information display.
        const zoomInfo = document.getElementById("zoomInfo");
        if (zoomInfo) {
            zoomInfo.textContent = `Zoom: ${Math.round(zoomLevel * 100)}%`;
        }
    }
    
    // Added event listeners to the buttons for modifying the annotation mode.
    lineBtn?.addEventListener("click", () => setMode("line"));
    dotBtn?.addEventListener("click", () => setMode("dot"));
    eraserBtn?.addEventListener("click", () => setMode("eraser"));
    eraseAllBtn?.addEventListener("click", () => setMode("eraseAll"));
    
    // Added event listeners for the zoom buttons as well.
    zoomInBtn.addEventListener("click", function() {
        if (zoomMode) {
            const imageCoords = screenToImageCoords(mouseX, mouseY);
            zoomToPoint(imageCoords.x, imageCoords.y);
        } else {
            // When activating zoom mode, deselect fill tool
            if (isFillToolActive) {
                isFillToolActive = false;
                const fillBtn = document.getElementById('fillToolBtn');
                if (fillBtn) {
                    fillBtn.classList.remove('btn-danger');
                    fillBtn.classList.add('btn-outline-danger');
                }
                const fillStatus = document.getElementById('fillToolStatus');
                if (fillStatus) {
                    fillStatus.style.display = 'none';
                }
                resetFillTool();
            }
            
            toggleZoomMode();
        }
    });

    zoomOutBtn?.addEventListener("click", function() {
        zoomOut();
    });

    resetZoomBtn?.addEventListener("click", function() {
        resetZoom();
    });

    if (img && annotationCanvas && annotationCtx) {
        // I added a mousedown, mousemove, mouseup and mouseleave listener to the canvas
        // so that it looks out for scribbling events and sets the annotations array accordingly.

        // mousedown: a click, mousemove: when you move your mouse, mouseup: when you let go of the mouse, mouseleave: when you leave
        // the component with the cursor.
        annotationCanvas.addEventListener("mousedown", (e) => {
            // This is when the user clicks on a certain location to zoom into. Here, this point will be used
            // as the center of the new region.
            if (zoomMode) {
                const imageCoords = screenToImageCoords(e.clientX, e.clientY);
                zoomToPoint(imageCoords.x, imageCoords.y);
                return;
            }
            
            const imageCoords = screenToImageCoords(e.clientX, e.clientY);
            mouseX = imageCoords.x;
            mouseY = imageCoords.y;

            if (isFillToolActive) {
                handleFillToolClick(imageCoords);
            }
        
            if (mode === "dot") {
                const layerId = createLayer("Dot", selectedColor);
                scribbles.push({ 
                    points: [{ x: mouseX, y: mouseY }], 
                    isPrediction: false,
                    color: selectedColor,
                    layerId: layerId
                });
                redrawAnnotations();
            } else if (mode === "line") {
                isDrawing = true;
                const layerId = createLayer("Line", selectedColor);
                currentStroke = [{ x: mouseX, y: mouseY }];
                currentStrokeColor = selectedColor;
                currentLayerId = layerId;
            } else if (mode === "eraser") {
                isErasing = true;
                eraseAtPoint(mouseX, mouseY);
            }
        });

        annotationCanvas.addEventListener("mousemove", (e) => {
            const imageCoords = screenToImageCoords(e.clientX, e.clientY);
            mouseX = imageCoords.x;
            mouseY = imageCoords.y;

            if (isFillToolActive && isDrawingBoundary) {
                currentBoundary.push({ x: mouseX, y: mouseY });
                redrawAnnotations();
            }

            if (isDrawing && mode === "line") {
                currentStroke.push({ x: mouseX, y: mouseY });
                redrawAnnotations();
            }
        
            if (mode === "eraser") {
                if (isErasing) {
                    eraseAtPoint(mouseX, mouseY);
                } else {
                    drawEraserCursor();
                }
            }
        });

        annotationCanvas.addEventListener("mouseup", () => {
            if (isDrawing && currentStroke.length > 0) {
                scribbles.push({ 
                    points: currentStroke, 
                    isPrediction: false,
                    color: currentStrokeColor,
                    layerId: currentLayerId
                });
                currentStroke = [];
                redrawAnnotations();
            }
            if (isFillToolActive && isDrawingBoundary) {
                isDrawingBoundary = false;
                
                // Close the boundary if we have at least 3 points
                if (currentBoundary.length >= 3) {
                    // Add the first point again to close the loop
                    currentBoundary.push({ ...currentBoundary[0] });
                    boundaryComplete = true;
                    updateFillToolStatus("Now click inside the boundary to fill it");
                } else {
                    // Not enough points for a valid boundary
                    currentBoundary = [];
                    updateFillToolStatus("Please draw a closed boundary with at least 3 points");
                }
                
                redrawAnnotations();
            }
            isDrawing = false;
            isErasing = false;
        });

        annotationCanvas.addEventListener("mouseleave", () => {
            if (isDrawing && currentStroke.length > 0) {
                scribbles.push({ 
                    points: currentStroke, 
                    isPrediction: false,
                    color: currentStrokeColor 
                });
                currentStroke = [];
                redrawAnnotations();
            }
            isDrawing = false;
            isErasing = false;
        });
    }

    function toggleFillTool() {
        // Reset other tools if active
        if (mode !== null) {
            mode = null;
            updateButtonStyles();
        }
        
        // Exit zoom mode if active
        if (zoomMode) {
            zoomMode = false;
            const zoomInBtn = document.getElementById('zoomInBtn');
            if (zoomInBtn) {
                zoomInBtn.classList.remove("btn-primary");
                zoomInBtn.classList.add("btn-outline-primary");
            }
            annotationCanvas.classList.remove("zoom-cursor");
            annotationCanvas.style.cursor = "crosshair";
        }
        
        // Toggle fill tool
        isFillToolActive = !isFillToolActive;
        
        // Update UI
        const fillBtn = document.getElementById('fillToolBtn');
        if (fillBtn) {
            if (isFillToolActive) {
                fillBtn.classList.remove('btn-outline-danger');
                fillBtn.classList.add('btn-danger');
                resetFillTool();
                updateFillToolStatus("Draw a closed boundary around the area you want to fill");
            } else {
                fillBtn.classList.remove('btn-danger');
                fillBtn.classList.add('btn-outline-danger');
                const status = document.getElementById('fillToolStatus');
                if (status) {
                    status.style.display = 'none';
                }
            }
        }
        
        // Update button styles
        updateButtonStyles();
    }
    
    // Reset fill tool state
    function resetFillTool() {
        isDrawingBoundary = false;
        currentBoundary = [];
        boundaryComplete = false;
    }
    
    // Update the status message
    function updateFillToolStatus(message) {
        const status = document.getElementById('fillToolStatus');
        if (status) {
            status.textContent = message;
            status.style.display = 'block';
        }
    }
    
    function handleFillToolClick(imageCoords) {
        if (!boundaryComplete) {
            isDrawingBoundary = true;
            currentBoundary = [{ x: imageCoords.x, y: imageCoords.y }];
        } else {
            const fillPoint = { x: imageCoords.x, y: imageCoords.y };
            
            // Check if click is inside the boundary
            if (isPointInPolygon(fillPoint, currentBoundary)) {
                // Create layer for the filled region
                const layerId = createLayer("Fill", selectedColor);
                
                // Add boundary to scribbles
                scribbles.push({
                    points: [...currentBoundary],
                    isPrediction: false,
                    color: selectedColor,
                    layerId: layerId
                });
                
                // Show loading message
                updateFillToolStatus("Filling region with dots... please wait");
                
                // Use setTimeout to allow the UI to update before doing the fill
                setTimeout(() => {
                    // Get fill points
                    const fillPoints = floodFill(fillPoint);
                    
                    // Add each dot as a separate stroke with a single point
                    // This matches your existing annotation system for dots
                    fillPoints.forEach(point => {
                        scribbles.push({
                            points: [point], // Single dot
                            isPrediction: false,
                            color: selectedColor,
                            layerId: layerId
                        });
                    });
                    
                    // Reset the fill tool
                    resetFillTool();
                    updateFillToolStatus("Fill complete. Draw another boundary or switch tools.");
                    redrawAnnotations();
                }, 50);
            } else {
                updateFillToolStatus("Click must be inside the boundary. Try again.");
            }
        }
    }
    
    // Check if a point is inside a polygon
    function isPointInPolygon(point, polygon) {
        // Using ray casting algorithm to determine if point is in polygon
        let inside = false;
        for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
            const xi = polygon[i].x, yi = polygon[i].y;
            const xj = polygon[j].x, yj = polygon[j].y;
            
            const intersect = ((yi > point.y) !== (yj > point.y)) &&
                (point.x < (xj - xi) * (point.y - yi) / (yj - yi) + xi);
                
            if (intersect) inside = !inside;
        }
        return inside;
    }
    
    // Flood fill algorithm (simplified for the canvas)
    function floodFill(fillPoint) {
        // Create a temporary canvas to perform the fill
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = annotationCanvas.width;
        tempCanvas.height = annotationCanvas.height;
        const tempCtx = tempCanvas.getContext('2d');
        
        // Apply the same zoom transformations to the temp canvas
        const offsetX = viewportCenterX - (originalImageDimensions.width / (2 * zoomLevel));
        const offsetY = viewportCenterY - (originalImageDimensions.height / (2 * zoomLevel));
        
        // Setup the transform to account for zoom (matching the main canvas)
        tempCtx.save();
        tempCtx.translate(-offsetX * zoomLevel, -offsetY * zoomLevel);
        tempCtx.scale(zoomLevel, zoomLevel);
        
        // Draw the boundary on the temp canvas with the same transformations as the main canvas
        tempCtx.beginPath();
        tempCtx.moveTo(currentBoundary[0].x, currentBoundary[0].y);
        for (let i = 1; i < currentBoundary.length; i++) {
            tempCtx.lineTo(currentBoundary[i].x, currentBoundary[i].y);
        }
        tempCtx.closePath();
        tempCtx.fill();
        tempCtx.restore();
        
        // Get the image data to find the filled pixels
        const imageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
        const data = imageData.data;
        
        // Create an array to hold filled points
        const filledPoints = [];
        
        // Sample points at a specific interval to create a grid of dots
        // Adjust sampleRate to control dot density
        const sampleRate = 5;
        
        for (let y = 0; y < tempCanvas.height; y += sampleRate) {
            for (let x = 0; x < tempCanvas.width; x += sampleRate) {
                const index = (y * tempCanvas.width + x) * 4; // RGBA format
                
                // If pixel is filled (non-zero alpha channel)
                if (data[index + 3] > 0) {
                    // Convert canvas coordinates back to image coordinates
                    const imageCoords = screenToImageCoords(x + annotationCanvas.getBoundingClientRect().left, 
                                                          y + annotationCanvas.getBoundingClientRect().top);
                    filledPoints.push(imageCoords);
                }
            }
        }
        
        return filledPoints;
    }

    // Main function to redraw all annotations with proper zoom and color
    redrawAnnotations = function() {
        // Call the original function to clear and setup
        annotationCtx.clearRect(0, 0, annotationCanvas.width, annotationCanvas.height);
        predictionCtx.clearRect(0, 0, predictionCanvas.width, predictionCanvas.height);
        
        const adjustForZoom = (context) => {
            context.save();
            
            const offsetX = viewportCenterX - (originalImageDimensions.width / (2 * zoomLevel));
            const offsetY = viewportCenterY - (originalImageDimensions.height / (2 * zoomLevel));
            
            // First, translate to adjust for pan.
            context.translate(-offsetX * zoomLevel, -offsetY * zoomLevel);
            
            // Then, scale for zoom.
            context.scale(zoomLevel, zoomLevel);
        };
        
        // Draw all of the user's annotations that are NOT predictions and are in visible layers
        for (const stroke of scribbles.filter(s => !s.isPrediction && (!s.layerId || visibleLayerIds.includes(s.layerId)))) {
            adjustForZoom(annotationCtx);
            
            // Handle fill batches more efficiently
            if (stroke.isFillBatch) {
                annotationCtx.fillStyle = stroke.color || "red";
                for (const point of stroke.points) {
                    annotationCtx.beginPath();
                    annotationCtx.arc(point.x, point.y, 1, 0, 2 * Math.PI);
                    annotationCtx.fill();
                }
            }
            // A simple dot
            else if (stroke.points.length === 1) {
                annotationCtx.fillStyle = stroke.color || "red";
                annotationCtx.beginPath();
                annotationCtx.arc(stroke.points[0].x, stroke.points[0].y, 2, 0, 2 * Math.PI);
                annotationCtx.fill();
            } 
            // A line/polygon
            else {
                annotationCtx.strokeStyle = stroke.color || "red";
                annotationCtx.lineWidth = 2 / zoomLevel;
                annotationCtx.beginPath();
                annotationCtx.moveTo(stroke.points[0].x, stroke.points[0].y);
                for (let i = 1; i < stroke.points.length; i++) {
                    annotationCtx.lineTo(stroke.points[i].x, stroke.points[i].y);
                }
                annotationCtx.stroke();
            }
            
            annotationCtx.restore();
        }
        
        // Code for prediction strokes - only draw if showPredictions is true
        if (showPredictions) {
            const predictionStrokes = scribbles.filter(s => s.isPrediction);
            if (predictionStrokes.length > 0) {
                adjustForZoom(predictionCtx);
                
                predictionCtx.strokeStyle = "blue";
                predictionCtx.lineWidth = 2 / zoomLevel;
                
                // I draw every contour point by point as it doesn't work well as a line.
                for (const stroke of predictionStrokes) {
                    if (stroke.points.length > 0) {
                        for (const point of stroke.points) {
                            predictionCtx.fillStyle = (point.color) ? point.color : (stroke.color || "blue");
                            predictionCtx.beginPath();
                            predictionCtx.arc(point.x, point.y, 2 / zoomLevel, 0, 2 * Math.PI);
                            predictionCtx.fill();
                        }
                    }
                }
                
                predictionCtx.restore();
            }
        }
        
        // If we're currently drawing, draw the current stroke
        if (isDrawing && currentStroke.length > 0) {
            adjustForZoom(annotationCtx);
            
            annotationCtx.strokeStyle = currentStrokeColor;
            annotationCtx.lineWidth = 2 / zoomLevel;
            annotationCtx.beginPath();
            annotationCtx.moveTo(currentStroke[0].x, currentStroke[0].y);
            for (let i = 1; i < currentStroke.length; i++) {
                annotationCtx.lineTo(currentStroke[i].x, currentStroke[i].y);
            }
            annotationCtx.stroke();
            
            annotationCtx.restore();
        }
    
        // Draw the eraser cursor.
        if (showEraserCursor && mode === "eraser") {
            adjustForZoom(annotationCtx);
            
            annotationCtx.beginPath();
            annotationCtx.strokeStyle = "rgba(0, 255, 255, 0.9)";
            annotationCtx.lineWidth = 1 / zoomLevel; 
            annotationCtx.arc(mouseX, mouseY, ERASE_RADIUS, 0, 2 * Math.PI);
            annotationCtx.stroke();
            
            annotationCtx.restore();
        }
        
        // Draw the current boundary if in fill tool mode
        if (isFillToolActive && currentBoundary.length > 1) {
            adjustForZoom(annotationCtx);
            
            annotationCtx.strokeStyle = selectedColor;
            annotationCtx.lineWidth = 2 / zoomLevel;
            
            // Draw the boundary
            annotationCtx.beginPath();
            annotationCtx.moveTo(currentBoundary[0].x, currentBoundary[0].y);
            for (let i = 1; i < currentBoundary.length; i++) {
                annotationCtx.lineTo(currentBoundary[i].x, currentBoundary[i].y);
            }
            
            if (boundaryComplete) {
                // If boundary is complete, close it
                annotationCtx.closePath();
            }
            
            annotationCtx.stroke();
            annotationCtx.restore();
        }
    }

    window.handleAnnotations = function () {
        // Ensure that the user's annotations are SAVED!! We need
        // the user to continuously modify their segmentation output.
        scribbles = scribbles.filter(s => !s.isPrediction);
        redrawAnnotations();

        // Redraws only the user annotations.
        const allPoints = scribbles
        .filter(s => !s.isPrediction)
        .flatMap(s => s.points.map(p => ({
            x: p.x,
            y: p.y,
            color: s.color // Keep the original color from the layer
        })));
        const annotationsJson = {
            image_name: window.imageName,
            shapes: [{
                label: "anomaly",
                points: allPoints.map(p => [p.x, p.y]),
                color: allPoints.map(p => p.color) // Send the original colors
            }]
        };

        fetch("/segment/", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "X-CSRFToken": getCookie("csrftoken")
            },
            body: JSON.stringify(annotationsJson)
        })
        .then(response => {
            if (!response.ok) {
                console.error("Server error:", response.status);
                return;
            }
            return response.json();
        })
        .then(data => {
            // Reset zoom when showing results
            zoomLevel = 1;
            viewportCenterX = originalImageDimensions.width / 2;
            viewportCenterY = originalImageDimensions.height / 2;
            
            const canvasWidth = annotationCanvas.width;
            const canvasHeight = annotationCanvas.height;
            
            // Keep canvas dimensions after processing
            const updateCanvasesButPreserveDimensions = () => {
                for (const c of [annotationCanvas, predictionCanvas]) {
                    c.width = canvasWidth;
                    c.height = canvasHeight;
                    c.style.width = `${canvasWidth}px`;
                    c.style.height = `${canvasHeight}px`;
                }
            };
            
            updateCanvasesButPreserveDimensions();
            
            // Remove previous predictions before adding new ones.
            scribbles = scribbles.filter(s => !s.isPrediction);

            const predictedPoints = data.predicted_annotations || [];
            if (predictedPoints.length > 0) {
                let stroke = [];
                let strokeColors = []; // Array to store individual point colors (Points here because adding the prediction contours point by point is easier.)
                for (const point of predictedPoints) {
                    if (Array.isArray(point[0])) {
                        const [[x, y], color] = point;
                        stroke.push({ x, y });
                        strokeColors.push(color);
                    } else {
                        const [x, y] = point;
                        stroke.push({ x, y });
                        strokeColors.push("blue"); // Default color if no color provided (Never happens but still.)
                    }
                }
                const pointsWithColors = stroke.map((point, index) => ({
                    x: point.x, 
                    y: point.y,
                    color: strokeColors[index] // Assign the color to each point
                }));
                
                scribbles.push({ 
                    points: pointsWithColors, 
                    isPrediction: true,
                    color: "blue" // This will be never used but it needs a color so just added blue.
                });
            }
            
            updateZoom();
            
            // Update zoom info
            const zoomInfo = document.getElementById("zoomInfo");
            if (zoomInfo) {
                zoomInfo.textContent = "Zoom: 100%";
            }
            
            const segmentedImageElement = document.getElementById("segmentedResultImage");
            segmentedImageElement.src = `data:image/png;base64,${data.segmented_image}`;
            if (data.segmented_image) {
                document.getElementById("segmentationResult").style.display = "block";

                // Download button
                const downloadBtn = document.getElementById("downloadSegmentedImage");
                downloadBtn.href = segmentedImageElement.src;
                downloadBtn.download = "segmented_result.png";
                downloadBtn.style.display = "inline-block";
            }
        });
    }

    window.handlePreprocessedImg = function () {
        const requestData = {
            image_name: window.imageName
        };

        fetch("/preprocessed-image/", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "X-CSRFToken": getCookie("csrftoken")
            },
            body: JSON.stringify(requestData)
        })
            .then(response => {
                if (!response.ok) {
                    throw new Error("Failed to fetch preprocessed image.");
                }
                return response.json();
            })
            .then(data => {
                if (data && data.preprocessed_image) {
                    const popup = document.createElement("div");
                    popup.style.position = "fixed";
                    popup.style.top = "50%";
                    popup.style.left = "50%";
                    popup.style.transform = "translate(-50%, -50%)";
                    popup.style.backgroundColor = "#fff";
                    popup.style.padding = "10px";
                    popup.style.border = "2px solid #444";
                    popup.style.zIndex = "9999";
                    popup.style.boxShadow = "0px 0px 10px rgba(0,0,0,0.5)";

                    const img = document.createElement("img");
                    img.src = `data:image/png;base64,${data.preprocessed_image}`;
                    img.style.maxWidth = "100%";
                    img.style.maxHeight = "80vh";

                    const closeButton = document.createElement("button");
                    closeButton.textContent = "Close";
                    closeButton.className = "btn btn-danger mt-3";
                    closeButton.onclick = () => popup.remove();

                    popup.appendChild(img);
                    popup.appendChild(closeButton);
                    document.body.appendChild(popup);
                } else {
                    alert("No preprocessed image returned.");
                }
            })
            .catch(error => {
                console.error("Error:", error);
                alert("Error displaying preprocessed image.");
            });
    }

    function getCookie(name) {
        const value = `; ${document.cookie}`;
        const parts = value.split(`; ${name}=`);
        return parts.length === 2 ? parts.pop().split(';').shift() : '';
    }

    function createLayer(type, color) {
        const layerId = `layer-${layerCounter++}`;
        const layerContainer = document.getElementById('layersContainer');
        
        const layerDiv = document.createElement('div');
        layerDiv.className = 'layer-item d-flex align-items-center justify-content-between p-2 rounded';
        layerDiv.style.width = '100%';
        layerDiv.id = layerId;
        layerDiv.setAttribute('data-layer-type', type);
        
        const layerName = document.createElement('input');
        layerName.type = 'text';
        layerName.value = `${type} ${layerCounter}`;
        layerName.style.color = 'var(--text-light)';
        layerName.style.backgroundColor = 'transparent';
        layerName.style.border = 'none';
        layerName.style.outline = 'none';
        layerName.style.width = '100px';
        layerName.addEventListener('change', function() {
            // Update the layer type tag when the name is changed
            layerDiv.setAttribute('data-layer-type', this.value.split(' ')[0]);
        });
        
        const layerControls = document.createElement('div');
        layerControls.className = 'd-flex align-items-center gap-2';
        
        const visibilityToggle = document.createElement('div');
        visibilityToggle.className = 'form-check form-switch';
        const visibilityInput = document.createElement('input');
        visibilityInput.className = 'form-check-input';
        visibilityInput.type = 'checkbox';
        visibilityInput.checked = true;
        visibilityInput.onchange = function() {
            toggleLayerVisibility(layerId);
        };
        visibilityToggle.appendChild(visibilityInput);
        
        const colorIndicator = document.createElement('div');
        colorIndicator.className = 'layer-color-indicator';
        colorIndicator.style.width = '20px';
        colorIndicator.style.height = '20px';
        colorIndicator.style.backgroundColor = color;
        colorIndicator.style.borderRadius = '4px';
        colorIndicator.style.border = '1px solid #ccc';
        colorIndicator.style.cursor = 'pointer';
        
        // Create a color picker input for the layer
        const layerColorPicker = document.createElement('input');
        layerColorPicker.type = 'color';
        layerColorPicker.value = color;
        layerColorPicker.style.width = '38px';
        layerColorPicker.style.height = '38px';
        layerColorPicker.style.padding = '0';
        layerColorPicker.style.border = 'none';
        layerColorPicker.style.borderRadius = '8px';
        layerColorPicker.style.cursor = 'pointer';
        layerColorPicker.style.position = 'fixed';
        layerColorPicker.style.opacity = '0';
        layerColorPicker.style.pointerEvents = 'none';
        
        // Create a container for the color picker to control its positioning
        const colorPickerContainer = document.createElement('div');
        colorPickerContainer.style.position = 'relative';
        colorPickerContainer.style.display = 'inline-block';
        colorPickerContainer.style.marginRight = '8px';
        colorPickerContainer.appendChild(colorIndicator);
        colorPickerContainer.appendChild(layerColorPicker);
        
        // Add click event to show color picker
        colorIndicator.addEventListener('click', function(e) {
            e.stopPropagation();
            const rect = colorIndicator.getBoundingClientRect();
            layerColorPicker.style.left = `${rect.left}px`;
            layerColorPicker.style.top = `${rect.bottom}px`;
            layerColorPicker.click();
        });
        
        // Handle color changes from the layer's color picker
        layerColorPicker.addEventListener('input', function(e) {
            const newColor = e.target.value;
            colorIndicator.style.backgroundColor = newColor;
            
            // Update all strokes in this layer with the new color
            scribbles.forEach(stroke => {
                if (stroke.layerId === layerId) {
                    stroke.color = newColor;
                }
            });
            
            redrawAnnotations();
        });
        
        // Add delete button with trash icon
        const deleteButton = document.createElement('button');
        deleteButton.innerHTML = '<i class="fa-solid fa-trash"></i>';
        deleteButton.className = 'btn btn-link p-0';
        deleteButton.style.color = '#6c757d';
        deleteButton.style.transition = 'color 0.2s';
        deleteButton.style.marginLeft = '8px';
        deleteButton.style.display = 'inline-flex';
        deleteButton.style.alignItems = 'center';
        deleteButton.style.justifyContent = 'center';
        deleteButton.querySelector('i').style.fontSize = '17px';
        deleteButton.querySelector('i').style.marginTop = '7px';
        deleteButton.onmouseover = function() {
            this.style.color = '#dc3545';
        };
        deleteButton.onmouseout = function() {
            this.style.color = '#6c757d';
        };
        deleteButton.onclick = function(e) {
            e.stopPropagation();
            // Remove the layer from DOM
            layerDiv.remove();
            // Remove all strokes associated with this layer
            scribbles = scribbles.filter(stroke => stroke.layerId !== layerId);
            // Remove from visible layers if present
            visibleLayerIds = visibleLayerIds.filter(id => id !== layerId);
            redrawAnnotations();
        };
        
        layerControls.appendChild(visibilityToggle);
        layerControls.appendChild(colorPickerContainer);
        layerControls.appendChild(deleteButton);
        
        layerDiv.appendChild(layerName);
        layerDiv.appendChild(layerControls);
        layerContainer.appendChild(layerDiv);
        
        // Add the new layer ID to visibleLayerIds by default
        if (!visibleLayerIds.includes(layerId)) {
            visibleLayerIds.push(layerId);
        }
        
        return layerId;
    }

    function toggleLayerVisibility(layerId) {
        const layer = document.getElementById(layerId);
        const isVisible = layer.querySelector('input[type="checkbox"]').checked;
        
        // Here we implement the actual visibility toggling of the annotation
        if (isVisible) {
            if (!visibleLayerIds.includes(layerId)) {
                visibleLayerIds.push(layerId);
            }
        } else {
            visibleLayerIds = visibleLayerIds.filter(id => id !== layerId);
        }
        redrawAnnotations();
    }

    // Add a function to toggle predictions visibility
    window.togglePredictions = function() {
        showPredictions = !showPredictions;
        redrawAnnotations();
    }

    // Function to erase all annotations
    function eraseAllAnnotations() {
        // Remove all non-prediction strokes
        scribbles = scribbles.filter(stroke => stroke.isPrediction);
        
        // Remove all layer elements
        const layersContainer = document.getElementById('layersContainer');
        while (layersContainer.firstChild) {
            layersContainer.removeChild(layersContainer.firstChild);
        }
        
        // Reset layer-related variables
        layerCounter = 0;
        visibleLayerIds = [];
        currentLayerId = null;
        
        // Redraw annotations (which will now be empty)
        redrawAnnotations();
    }

    //Keyboard shortcuts
    document.addEventListener('keydown', function (event) {
        //won't listen if user is typing
        if(["INPUT", "TEXTAREA"].includes(document.activeElement.tagName)) {
            return;
        }

        const key = event.key.toUpperCase();

        if (key === "L") {
            setMode(mode ==="line" ? null: "line");
        }
        else if (key === "E") {
            setMode(mode ==="eraser" ? null: "eraser");
        }
        else if (key === "D") {
            setMode(mode ==="dot" ? null: "dot");
        }
        else if (key === "A") {
            setMode(mode ==="eraseAll" ? null: "eraseAll");
        }
        else if (key === "Z") {
            toggleZoomMode();

            if (zoomMode) {
                zoomInBtn.classList.add("active");
                zoomInBtn.classList.remove("btun-outline-primary");
                zoomInBtn.classList.add("btn-primary");
            }
            else {
                zoomInBtn.classList.remove("active");
                zoomInBtn.classList.add("btun-outline-primary");
                zoomInBtn.classList.remove("btn-primary");
            }

        }
    });
});