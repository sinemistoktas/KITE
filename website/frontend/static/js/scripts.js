/*!
•⁠  ⁠Start Bootstrap - Resume v7.0.6 (https://startbootstrap.com/theme/resume)
•⁠  ⁠Copyright 2013-2023 Start Bootstrap
•⁠  ⁠Licensed under MIT (https://github.com/StartBootstrap/startbootstrap-resume/blob/master/LICENSE)
*/
//
// Scripts
// 

let mouseX = 0;
let mouseY = 0;
let showEraserCursor = false;
let mode = null; 
let isDrawing = false;
let isErasing = false;
const ERASE_RADIUS = 10;
let annotationCanvas, annotationCtx, predictionCanvas, predictionCtx;

// GLOBAL references to be set on DOMContentLoaded
let canvas, ctx, scribbles = [], currentStroke = [];

// Useful for the eraser feature.
function distance(p1, p2) { 
    return Math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2);
}


window.addEventListener('DOMContentLoaded', event => {

    annotationCanvas = document.getElementById("annotationCanvas");
    annotationCtx = annotationCanvas.getContext("2d");

    predictionCanvas = document.getElementById("predictionCanvas");
    predictionCtx = predictionCanvas.getContext("2d");

    let selectedColor = "#ff0000"; // def annotation color is red
    const colorPicker = document.getElementById("colorPicker");
    if (colorPicker) {
        colorPicker.addEventListener("input", (e) => {
            selectedColor = e.target.value;
        });
    }

    function drawLine(points, color = selectedColor, context = annotationCtx) {
        if (points.length < 2) return;
        context.strokeStyle = color;
        context.lineWidth = 2;
        context.beginPath();
        context.moveTo(points[0].x, points[0].y);
        for (let i = 1; i < points.length; i++) {
            context.lineTo(points[i].x, points[i].y);
        }
        context.stroke();
    }

    function eraseAtPoint(x, y) {
        scribbles = scribbles.filter(stroke => {
            if (stroke.isPrediction) return true;
            return !stroke.points.some(p => distance(p, { x, y }) < ERASE_RADIUS);
        });
    
        annotationCtx.clearRect(0, 0, annotationCanvas.width, annotationCanvas.height);
        for (const stroke of scribbles) {
            if (!stroke.isPrediction) {
                const color = stroke.color;
                if (stroke.points.length === 1) {
                    annotationCtx.fillStyle = color;
                    annotationCtx.beginPath();
                    annotationCtx.arc(stroke.points[0].x, stroke.points[0].y, 2, 0, 2 * Math.PI);
                    annotationCtx.fill();
                } else {
                    drawLine(stroke.points, color);
                }
            }
        }
    }

    function drawEraserCursor() {
        if (!showEraserCursor || mode !== "eraser") return;
    
        annotationCtx.clearRect(0, 0, annotationCanvas.width, annotationCanvas.height);
        for (const stroke of scribbles) {
            if (!stroke.isPrediction) {
                const color = stroke.color ;
                if (stroke.points.length === 1) {
                    annotationCtx.fillStyle = color;
                    annotationCtx.beginPath();
                    annotationCtx.arc(stroke.points[0].x, stroke.points[0].y, 2, 0, 2 * Math.PI);
                    annotationCtx.fill();
                } else {
                    drawLine(stroke.points, color);
                }
            }
        }
    
        annotationCtx.save();
        annotationCtx.beginPath();
        annotationCtx.strokeStyle = "rgba(0, 255, 255, 0.9)";
        annotationCtx.lineWidth = 1;
        annotationCtx.arc(mouseX, mouseY, ERASE_RADIUS, 0, 2 * Math.PI);
        annotationCtx.stroke();
        annotationCtx.restore();
    }

    const img = document.getElementById("uploadedImage");
    canvas = document.getElementById("annotationCanvas");
    ctx = canvas?.getContext("2d");

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
    }

    // A more generic function to set the mode rather than the previous calls. This ensures that a button does not need to be
    // clicked again to reset its mode's properties, like the moving eraser cursor.

    function setMode(newMode) {
        if (mode === newMode) {
            // Deselect if user clicks the same button again.
            mode = null;
            showEraserCursor = false;
        } else {
            const previousMode = mode;
            mode = newMode;
            showEraserCursor = (newMode === "eraser");
    
            // If switching FROM eraser mode, clear the eraser cursor from canvas.
            if (previousMode === "eraser" && newMode !== "eraser") {
                annotationCanvas.style.cursor = "crosshair";
            
                annotationCtx.clearRect(0, 0, annotationCanvas.width, annotationCanvas.height);
                for (const stroke of scribbles) {
                    if (!stroke.isPrediction) {
                        // Fix: Use stroke.color instead of undefined color variable
                        if (stroke.points.length === 1) {
                            annotationCtx.fillStyle = stroke.color;
                            annotationCtx.beginPath();
                            annotationCtx.arc(stroke.points[0].x, stroke.points[0].y, 2, 0, 2 * Math.PI);
                            annotationCtx.fill();
                        } else {
                            drawLine(stroke.points, stroke.color);
                        }
                    }
                }
            }
            // Update cursor visibility based on mode.
            if (newMode === "eraser") {
                canvas.style.cursor = "none";
            } else {
                canvas.style.cursor = "crosshair";
            }
        }
    
        updateButtonStyles();
    }
    
    // Added event listeners to the buttons for modifying the annotation mode.
    lineBtn?.addEventListener("click", () => setMode("line"));
    dotBtn?.addEventListener("click", () => setMode("dot"));
    eraserBtn?.addEventListener("click", () => setMode("eraser"));


    if (img && canvas && ctx) {
        // I added a mousedown, mousemove, mouseup and mouseleave listener to the canvas
        // so that it looks out for scribbling events and sets the annotations array accordingly.
        canvas.addEventListener("mousedown", (e) => {
            const rect = canvas.getBoundingClientRect();
            const x = Math.round(e.clientX - rect.left);
            const y = Math.round(e.clientY - rect.top);

            if (mode === "dot") {
                ctx.fillStyle = selectedColor;
                ctx.beginPath();
                ctx.arc(x, y, 2, 0, 2 * Math.PI);
                ctx.fill();
                scribbles.push({ points: [{ x, y }], isPrediction: false , color: selectedColor});
            } else if (mode === "line") {
                isDrawing = true;
                currentStroke = [{ x, y }];
                currentStrokeColor = selectedColor; 
            } else if (mode === "eraser") {
                isErasing = true;
                eraseAtPoint(x, y);
            }
        });

        canvas.addEventListener("mousemove", (e) => {
            const rect = canvas.getBoundingClientRect();
            mouseX = Math.round(e.clientX - rect.left);
            mouseY = Math.round(e.clientY - rect.top);
        
            if (isDrawing && mode === "line") { 
                currentStroke.push({ x: mouseX, y: mouseY });
                // Fix: Use selectedColor directly rather than color=selectedColor which causes syntax error
                drawLine(currentStroke, selectedColor);
            }
        
            if (mode === "eraser") {
                if (isErasing) {
                    eraseAtPoint(mouseX, mouseY);
                } else {
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    for (const stroke of scribbles) {
                        // Fix: Check for points array and use stroke.color
                        if (stroke.points && stroke.points.length === 1) {
                            ctx.fillStyle = stroke.color || "red";
                            ctx.beginPath();
                            ctx.arc(stroke.points[0].x, stroke.points[0].y, 2, 0, 2 * Math.PI);
                            ctx.fill();
                        } else if (stroke.points) {
                            // Fix: Use proper color handling
                            const strokeColor = stroke.isPrediction ? "blue" : (stroke.color || "red");
                            drawLine(stroke.points, strokeColor);
                        }
                    }
                }
        
                drawEraserCursor();
            }
        });

        canvas.addEventListener("mouseup", () => {
            if (isDrawing && currentStroke.length > 0) {
                scribbles.push({ points: currentStroke, isPrediction: false , color: selectedColor});
                currentStroke = [];
            }
            isDrawing = false;
            isErasing = false;
        });

        canvas.addEventListener("mouseleave", () => {
            isDrawing = false;
            isErasing = false;
        });
    }

    window.handleAnnotations = function () {
        // Ensure that the user's annotations are SAVED!! We need
        // the user to continuously modify their segmentation output.
        scribbles = scribbles.filter(s => !s.isPrediction);
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Redraws only the user annotations.
        for (const stroke of scribbles) {
            const color = stroke.color || "red";
            if (stroke.points.length === 1) {
                ctx.fillStyle = color;
                ctx.beginPath();
                ctx.arc(stroke.points[0].x, stroke.points[0].y, 2, 0, 2 * Math.PI);
                ctx.fill();
            } else {
                drawLine(stroke.points, color);
            }
        }
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
            const img = document.getElementById("uploadedImage");
            const canvas = document.getElementById("annotationCanvas");
            const ctx = canvas.getContext("2d");

            resizeCanvasToImage();
            predictionCtx.clearRect(0, 0, predictionCanvas.width, predictionCanvas.height);

            scribbles = scribbles.filter(s => !s.isPrediction);

            for (const stroke of scribbles) {
                const color = stroke.color ;
                if (stroke.points.length === 1) {
                    annotationCtx.fillStyle = color;
                    annotationCtx.beginPath();
                    annotationCtx.arc(stroke.points[0].x, stroke.points[0].y, 2, 0, 2 * Math.PI);
                    annotationCtx.fill();
                } else {
                    drawLine(stroke.points, color);
                }
            }

            const predictedPoints = data.predicted_annotations || [];
            if (predictedPoints.length > 0) {
                let stroke = [];
                for (const [[x, y], color] of predictedPoints) {
                    predictionCtx.fillStyle = color;
                    predictionCtx.beginPath();
                    predictionCtx.arc(x, y, 2, 0, 2 * Math.PI);
                    predictionCtx.fill();
                    stroke.push({ x, y });
                }
                scribbles.push({ points: stroke, isPrediction: true , color: selectedColor});
            }
            const segmentedImageElement = document.getElementById("segmentedResultImage");
            const base64Image =`data:image/png;base64,${data.segmented_image}`;

            segmentedImageElement.src =`data:image/png;base64,${data.segmented_image}`;
            if (data.segmented_image) {
                document.getElementById("segmentationResult").style.display = "block";

                //download
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
                    img.src =`data:image/png;base64,${data.preprocessed_image}`;
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