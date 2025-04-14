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
    //TODO: When the window is resized the canvas is DELETED. what to do ?
      
    if (img.complete) { // If the browser has finished loading the image, resize the canvas. Else resize it when it loads.
        resizeCanvasToImage();
    } else {
        img.onload = () => {
            resizeCanvasToImage();
        };
    }

    window.addEventListener("resize", resizeCanvasToImage); // Whenever the browser window & the image is resized, the canvas would resize with it.

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
        // Added a click event listener to the canvas.
        canvas.addEventListener("mousedown", (e) => {
            isDrawing = true;
            currentStroke = [];
            const rect = canvas.getBoundingClientRect();
            currentStroke.push({
                x: Math.round(e.clientX - rect.left),
                y: Math.round(e.clientY - rect.top),
            });
        });

        canvas.addEventListener("mousemove", (e) => {
            if (!isDrawing) return;
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

    window.handleAnnotations = function () {
        const allPoints = scribbles.flat(); // Flattens all the scribbles into a single array.
        const annotationsJson = {
            image_name: window.imageName,
            shapes: [{
                label: "anomaly",
                points: annotations.map(p => [p.x, p.y])

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
        .then(response => response.blob())
        .then(blob => {
            const url = URL.createObjectURL(blob);
            const img = new Image();
            img.src = url;
            document.getElementById("segmentationResult").innerHTML = "";
            document.getElementById("segmentationResult").appendChild(img);
        });
    }

    function getCookie(name) {
        const value = `; ${document.cookie}`;
        const parts = value.split(`; ${name}=`);
        return parts.length === 2 ? parts.pop().split(';').shift() : '';
    }

});