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

// Useful for the eraser feature.
function distance(p1, p2) { 
    return Math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2);
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

    annotationCanvas = document.getElementById("annotationCanvas");
    annotationCtx = annotationCanvas.getContext("2d");

    predictionCanvas = document.getElementById("predictionCanvas");
    predictionCtx = predictionCanvas.getContext("2d");

    // Color picker functionality from first script
    const colorPicker = document.getElementById("colorPicker");
    if (colorPicker) {
        colorPicker.addEventListener("input", (e) => {
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
        scribbles = scribbles.filter(stroke => {
            if (stroke.isPrediction) return true; // if the line is the machine's predictions, DO NOT erase it.
            return !stroke.points.some(p => distance(p, { x, y }) < ERASE_RADIUS); // If the scribble has a point that
            // is within the radius of the eraser, delete it. 
        });
    
        redrawAnnotations(); // Rerender the annotations with the newly erased component.
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
        } else if (mode === "dot") {
            dotBtn.classList.remove("btn-outline-danger");
            dotBtn.classList.add("btn-danger");
            lineBtn.classList.remove("btn-danger");
            lineBtn.classList.add("btn-outline-danger");
            eraserBtn.classList.remove("btn-danger");
            eraserBtn.classList.add("btn-outline-danger");
        } else if (mode === "eraser") {
            eraserBtn.classList.remove("btn-outline-danger");
            eraserBtn.classList.add("btn-danger");
            lineBtn.classList.remove("btn-danger");
            lineBtn.classList.add("btn-outline-danger");
            dotBtn.classList.remove("btn-danger");
            dotBtn.classList.add("btn-outline-danger");
        }
        else { // Deselect all buttons if the user has clicked on a button that they have already clicked on.
            lineBtn.classList.remove("btn-danger");
            lineBtn.classList.add("btn-outline-danger");
            dotBtn.classList.remove("btn-danger");
            dotBtn.classList.add("btn-outline-danger");
            eraserBtn.classList.remove("btn-danger");
            eraserBtn.classList.add("btn-outline-danger");
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
        if (mode === newMode) {
            // Deselect if user clicks the same button again.
            mode = null;
            showEraserCursor = false;
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
            // Deactivate any other mode
            mode = null;
            showEraserCursor = false;
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
    
    // Added event listeners for the zoom buttons as well.
    zoomInBtn?.addEventListener("click", toggleZoomMode);
    zoomOutBtn?.addEventListener("click", zoomOut);
    resetZoomBtn?.addEventListener("click", resetZoom);

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
        
            if (mode === "dot") {
                scribbles.push({ 
                    points: [{ x: mouseX, y: mouseY }], 
                    isPrediction: false,
                    color: selectedColor 
                });
                redrawAnnotations();
            } else if (mode === "line") {
                isDrawing = true;
                currentStroke = [{ x: mouseX, y: mouseY }];
                currentStrokeColor = selectedColor;
            } else if (mode === "eraser") {
                isErasing = true;
                eraseAtPoint(mouseX, mouseY);
            }
        });

        annotationCanvas.addEventListener("mousemove", (e) => {
            const imageCoords = screenToImageCoords(e.clientX, e.clientY);
            mouseX = imageCoords.x;
            mouseY = imageCoords.y;
        
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
                    color: currentStrokeColor
                });
                currentStroke = [];
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

    // Main function to redraw all annotations with proper zoom and color
    function redrawAnnotations() {
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
        
        // Draw all of the user's annotations that are NOT predictions.
        for (const stroke of scribbles.filter(s => !s.isPrediction)) {
            adjustForZoom(annotationCtx);
            
            // A simple dot.
            if (stroke.points.length === 1) {
                annotationCtx.fillStyle = stroke.color || "red";
                annotationCtx.beginPath();
                annotationCtx.arc(stroke.points[0].x, stroke.points[0].y, 2, 0, 2 * Math.PI);
                annotationCtx.fill();
            } else { // Otherwise, a line.
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
        
        // Code for prediction strokes.
        const predictionStrokes = scribbles.filter(s => s.isPrediction);
        if (predictionStrokes.length > 0) {
            adjustForZoom(predictionCtx);
            
            predictionCtx.strokeStyle = "blue";
            predictionCtx.lineWidth = 2 / zoomLevel;
            
            // I draw every contour point by point as it doesn't work well as a line.
            for (const stroke of predictionStrokes) {
                if (stroke.points.length > 0) {
                    for (const point of stroke.points) {
                        predictionCtx.fillStyle = stroke.color || "blue";
                        predictionCtx.beginPath();
                        predictionCtx.arc(point.x, point.y, 2 / zoomLevel, 0, 2 * Math.PI);
                        predictionCtx.fill();
                    }
                }
            }
            
            predictionCtx.restore();
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
            color: s.color // add color from parent scribble
        })));
        const annotationsJson = {
            image_name: window.imageName,
            shapes: [{
                label: "anomaly",
                points: allPoints.map(p => [p.x, p.y]),
                color: allPoints.map(p => p.color)
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
                // Handle both formats: [[x,y], color] or [x,y]
                for (const point of predictedPoints) {
                    if (Array.isArray(point[0])) {
                        const [[x, y], color] = point;
                        stroke.push({ x, y });
                    } else {
                        const [x, y] = point;
                        stroke.push({ x, y });
                    }
                }
                scribbles.push({ 
                    points: stroke, 
                    isPrediction: true,
                    color: "blue" // Default prediction color
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
});