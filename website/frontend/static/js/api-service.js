import { state } from './state.js';
import { redrawAnnotations, resetZoom } from './canvas-tools.js';

export function getCSRFToken(name = "csrftoken") {
    const value = `; ${document.cookie}`;
    const parts = value.split(`; ${name}=`);
    return parts.length === 2 ? parts.pop().split(';').shift() : '';
}

export function handleAnnotations() {
    state.scribbles = state.scribbles.filter(s => !s.isPrediction);
    redrawAnnotations();

    const allPoints = state.scribbles.flatMap(s =>
        s.points.map(p => ({
            x: p.x,
            y: p.y,
            color: s.color
        }))
    );

    const payload = {
        image_name: state.imageName,
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
            "X-CSRFToken": getCSRFToken()
        },
        body: JSON.stringify(payload)
    })
    .then(res => res.json())
    .then(data => {
        resetZoom();

        const resultImage = document.getElementById("segmentedResultImage");
        resultImage.src = `data:image/png;base64,${data.segmented_image}`;

        document.getElementById("segmentationResult").style.display = "block";
        const downloadBtn = document.getElementById("downloadSegmentedImage");
        downloadBtn.href = resultImage.src;
        downloadBtn.download = "segmented_result.png";
        downloadBtn.style.display = "inline-block";


        
        const stageContainer = document.getElementById("segmentationStage");
        stageContainer.innerHTML = ""; // Clear previous content
      

        const width = resultImage.naturalWidth || resultImage.clientWidth;
        const height = resultImage.naturalHeight || resultImage.clientHeight;
        stageContainer.style.width = width + "px";
        stageContainer.style.height = height + "px";
            
        const stage = new Konva.Stage({
          container: "segmentationStage",
          width: width,
          height: height,
        });

        const layer = new Konva.Layer();
        stage.add(layer);

      
        (data.final_mask || []).forEach(({ regionId, pixels, color }) => {
            const group = new Konva.Group({
              id: regionId,
              draggable: true,
            });
          
            pixels.forEach(([y, x]) => {
              const rect = new Konva.Rect({
                x: x,
                y: y,
                width: 1,
                height: 1,
                fill: color || "rgba(233, 37, 37, 0.98)",
              });
              group.add(rect);
            });
          
            group.on("click", () => {
              console.log(`Clicked region: ${regionId}`);
            });
          
            layer.add(group);
          });
          
          layer.draw();
          
        
          

        const predictedPoints = data.predicted_annotations || [];
        const strokes = predictedPoints.map(p => {
            if (Array.isArray(p[0])) {
                const [[x, y], color] = p;
                return { x, y, color };
            } else {
                const [x, y] = p;
                return { x, y, color: "blue" };
            }
        });

        state.scribbles.push({
            points: strokes,
            isPrediction: true,
            color: "blue"
        });

        redrawAnnotations();
    });
}

export function handlePreprocessedImg() {
    fetch("/preprocessed-image/", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "X-CSRFToken": getCSRFToken()
        },
        body: JSON.stringify({ image_name: state.imageName })
    })
    .then(res => res.json())
    .then(data => {
        if (!data?.preprocessed_image) return alert("No preprocessed image returned.");

        const popup = document.createElement("div");
        popup.style = `
            position: fixed; top: 50%; left: 50%;
            transform: translate(-50%, -50%);
            background-color: #fff; padding: 10px;
            border: 2px solid #444; z-index: 9999;
            box-shadow: 0 0 10px rgba(0,0,0,0.5)
        `;

        const img = document.createElement("img");
        img.src = `data:image/png;base64,${data.preprocessed_image}`;
        img.style.maxWidth = "100%";
        img.style.maxHeight = "80vh";

        const close = document.createElement("button");
        close.className = "btn btn-danger mt-3";
        close.innerText = "Close";
        close.onclick = () => popup.remove();

        popup.append(img, close);
        document.body.appendChild(popup);
    });
}