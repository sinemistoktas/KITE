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
let mode = null; 
let isDrawing = false;
let isErasing = false;
const ERASE_RADIUS = 10;

// GLOBAL references to be set on DOMContentLoaded
let canvas, ctx, scribbles = [], currentStroke = [];

// Useful for the eraser feature.
function distance(p1, p2) { 
    return Math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2);
}


window.addEventListener('DOMContentLoaded', event => {

    function drawLine(points) {
        if (points.length < 2) return;
        ctx.strokeStyle = "red";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(points[0].x, points[0].y);
        for (let i = 1; i < points.length; i++) {
            ctx.lineTo(points[i].x, points[i].y);
        }
        ctx.stroke();
    }

    function eraseAtPoint(x, y) {
        scribbles = scribbles.filter(stroke => {
            return !stroke.some(p => distance(p, { x, y }) < ERASE_RADIUS);
        });
    
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        for (const stroke of scribbles) {
            if (stroke.length === 1) {
                ctx.fillStyle = "red";
                ctx.beginPath();
                ctx.arc(stroke[0].x, stroke[0].y, 2, 0, 2 * Math.PI);
                ctx.fill();
            } else {
                drawLine(stroke);
            }
        }
    }

    function drawEraserCursor() {
        if (!showEraserCursor || mode !== "eraser") return;

        ctx.save();
        ctx.beginPath();
        ctx.strokeStyle = "rgba(0, 0, 255, 0.4)";
        ctx.lineWidth = 1;
        ctx.arc(mouseX, mouseY, ERASE_RADIUS, 0, 2 * Math.PI);
        ctx.stroke();
        ctx.restore();
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

    // Sets the style of the canvas to ensure that it is completely aligned with the image.
    function resizeCanvasToImage() {
        const rect = img.getBoundingClientRect();
        canvas.width = rect.width;
        canvas.height = rect.height;
        canvas.style.width = `${rect.width}px`;
        canvas.style.height = `${rect.height}px`;
        canvas.style.position = "absolute";
        canvas.style.top = "0";
        canvas.style.left = "0";
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


     // Added event listeners to the buttons for modifying the annotation mode.

    lineBtn?.addEventListener("click", () => {
        mode = mode === "line" ? null : "line";
        updateButtonStyles();
    });
    
    dotBtn?.addEventListener("click", () => {
        mode = mode === "dot" ? null : "dot";
        updateButtonStyles();
    });

    eraserBtn?.addEventListener("click", () => {
        if (mode === "eraser") {
            mode = null;
            showEraserCursor = false;
        } else {
            mode = "eraser";
            showEraserCursor = true;
        }
        updateButtonStyles();
    });



    if (img && canvas && ctx) {
        // I added a mousedown, mousemove, mouseup and mouseleave listener to the canvas
        // so that it looks out for scribbling events and sets the annotations array accordingly.
        canvas.addEventListener("mousedown", (e) => {
            const rect = canvas.getBoundingClientRect();
            const x = Math.round(e.clientX - rect.left);
            const y = Math.round(e.clientY - rect.top);

            if (mode === "dot") {
                ctx.fillStyle = "red";
                ctx.beginPath();
                ctx.arc(x, y, 2, 0, 2 * Math.PI);
                ctx.fill();
                scribbles.push([{ x, y }]); // So with a dot, the annotation will simply be saved as a single point of x and y.
            } else if (mode === "line") {
                isDrawing = true; // The user has started drawing if the mode is the line mode.
                currentStroke = [{ x, y }];
            } else if (mode === "eraser") {
                isErasing = true;
                eraseAtPoint(x, y);
            }
        });

        canvas.addEventListener("mousemove", (e) => {
            const rect = canvas.getBoundingClientRect();
            mouseX = Math.round(e.clientX - rect.left);
            mouseY = Math.round(e.clientY - rect.top);
        
            if (isDrawing && mode === "line") { // Ensure that the user is either currently drawing AND we're in the line mode. 
                currentStroke.push({ x: mouseX, y: mouseY });
                drawLine(currentStroke);
            }
        
            if (mode === "eraser") {
                if (isErasing) {
                    eraseAtPoint(mouseX, mouseY);
                } else {
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    for (const stroke of scribbles) {
                        if (stroke.length === 1) {
                            ctx.fillStyle = "red";
                            ctx.beginPath();
                            ctx.arc(stroke[0].x, stroke[0].y, 2, 0, 2 * Math.PI);
                            ctx.fill();
                        } else {
                            drawLine(stroke);
                        }
                    }
                }
        
                drawEraserCursor();
            }
        });

        canvas.addEventListener("mouseup", () => {
            if (isDrawing && currentStroke.length > 0) {
                scribbles.push(currentStroke);
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
        const allPoints = scribbles.flat();
        const annotationsJson = {
            image_name: window.imageName,
            shapes: [{
                label: "anomaly",
                points: allPoints.map(p => [p.x, p.y])
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

            img.src = `data:image/png;base64,${data.segmented_image}`;
            img.onload = () => {
                resizeCanvasToImage();
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                scribbles = [];

                const predictedPoints = data.predicted_annotations || [];
                let stroke = [];

                for (const [x, y] of predictedPoints) {
                    ctx.fillStyle = "blue";
                    ctx.beginPath();
                    ctx.arc(x, y, 2, 0, 2 * Math.PI);
                    ctx.fill();
                    stroke.push({ x, y });
                }

                if (stroke.length > 0) {
                    scribbles.push(stroke);
                }
            };
        });
    }

    function getCookie(name) {
        const value = `; ${document.cookie}`;
        const parts = value.split(`; ${name}=`);
        return parts.length === 2 ? parts.pop().split(';').shift() : '';
    }
});