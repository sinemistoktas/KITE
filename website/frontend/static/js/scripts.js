/*!
* Start Bootstrap - Resume v7.0.6 (https://startbootstrap.com/theme/resume)
* Copyright 2013-2023 Start Bootstrap
* Licensed under MIT (https://github.com/StartBootstrap/startbootstrap-resume/blob/master/LICENSE)
*/
//
// Scripts
// 

window.addEventListener('DOMContentLoaded', event => {

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

    // Logic for image annotation
    const img = document.getElementById("uploadedImage");
    const canvas = document.getElementById("annotationCanvas");
    const ctx = canvas?.getContext("2d");
    let mode = "line"; // This will be the current mode of annotation for the user. For now, we have two tools: lines and dots.

    let scribbles = [];
    let currentStroke = [];
    let isDrawing = false;

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
      
    if (img.complete) { // If the browser has finished loading the image, resize the canvas. Else resize it when it loads.
        resizeCanvasToImage();
    } else {
        img.onload = () => {
            resizeCanvasToImage();
        };
    }

    window.addEventListener("resize", resizeCanvasToImage); // Whenever the browser window & the image is resized, the canvas would resize with it.

    // Added event listeners to the buttons for modifying the annotation mode.
    document.getElementById("scribbleMode").addEventListener("click", () => {
        mode = "line";
    });

    document.getElementById("dotMode").addEventListener("click", () => {
        mode = "dot";
    })

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
                scribbles.push([{x, y}]); // So with a dot, the annotation will simply be saved as a single point of x and y.
            } else {
                isDrawing = true // The user has started drawing if the mode is the line mode.
                currentStroke = [{x, y}]
            }
        });

        canvas.addEventListener("mousemove", (e) => {
            if (!isDrawing || mode !== "line") return; // Ensure that the user is either currently drawing AND we're in the line mode. 
            const rect = canvas.getBoundingClientRect();
            const x = Math.round(e.clientX - rect.left);
            const y = Math.round(e.clientY - rect.top);
            currentStroke.push({x, y});
            drawLine(currentStroke.slice(-2));
        });

        ["mouseup", "mouseleave"].forEach(eventName => {
            canvas.addEventListener(eventName, () => {
                if (isDrawing && currentStroke.length > 1) {
                    scribbles.push(currentStroke);
                }
                isDrawing = false;
            });
        });
    }
});