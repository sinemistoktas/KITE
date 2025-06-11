// website/frontend/static/js/layers.js

import { state } from './state.js';
import { redrawAnnotations } from './canvas-tools.js';

export function createLayer(type, color) {
    const layerId = `layer-${state.layerCounter++}`; // ✅ Unique ID
    const container = document.getElementById('layersContainer');

    const layerDiv = document.createElement('div');
    layerDiv.className = 'layer-item d-flex align-items-center justify-content-between p-2 rounded';
    layerDiv.style.width = '100%';
    layerDiv.id = layerId;
    layerDiv.setAttribute('data-layer-type', type);

    const layerName = document.createElement('input');
    layerName.type = 'text';
    layerName.value = `${type} ${state.layerCounter}`;
    layerName.style = `
        color: var(--text-light);
        background-color: transparent;
        border: none;
        outline: none;
        width: 100px;
    `;

    layerName.addEventListener('change', function () {
        layerDiv.setAttribute('data-layer-type', this.value.split(' ')[0]);
    });

    const layerControls = document.createElement('div');
    layerControls.className = 'd-flex align-items-center justify-content-end flex-wrap gap-1';

    const visibilityToggle = document.createElement('div');
    visibilityToggle.className = 'form-check form-switch';
    const visibilityInput = document.createElement('input');
    visibilityInput.className = 'form-check-input';
    visibilityInput.type = 'checkbox';
    visibilityInput.checked = true;
    visibilityInput.onchange = () => toggleLayerVisibility(layerId);
    visibilityToggle.appendChild(visibilityInput);

    const colorIndicator = document.createElement('div');
    colorIndicator.className = 'layer-color-indicator';
    colorIndicator.style = `
        width: 20px; height: 20px;
        background-color: ${color};
        border-radius: 4px;
        border: 1px solid #ccc;
        cursor: pointer;
    `;

    const colorPicker = document.createElement('input');
    colorPicker.type = 'color';
    colorPicker.value = color;
    colorPicker.style = `
        width: 38px;
        height: 38px;
        padding: 0;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        position: fixed;
        opacity: 0;
        pointer-events: none;
    `;

    const colorPickerContainer = document.createElement('div');
    colorPickerContainer.style = 'position: relative; display: inline-block; margin-right: 8px;';
    colorPickerContainer.appendChild(colorIndicator);
    colorPickerContainer.appendChild(colorPicker);

    colorIndicator.onclick = e => {
        e.stopPropagation();
        const rect = colorIndicator.getBoundingClientRect();
        colorPicker.style.left = `${rect.left}px`;
        colorPicker.style.top = `${rect.bottom}px`;
        colorPicker.click();
    };

    colorPicker.oninput = e => {
        const newColor = e.target.value;
        colorIndicator.style.backgroundColor = newColor;
        state.scribbles.forEach(stroke => {
            if (stroke.layerId === layerId) stroke.color = newColor;
        });
        redrawAnnotations();
    };

    const deleteBtn = document.createElement('button');
    deleteBtn.innerHTML = '<i class="fa-solid fa-trash"></i>';
    deleteBtn.className = 'btn btn-link p-0';
    deleteBtn.style = `
        color: #6c757d;
        transition: color 0.2s;
        margin-left: 8px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
    `;
    deleteBtn.querySelector('i').style = 'font-size: 17px; margin-top: 7px;';
    deleteBtn.onmouseover = () => (deleteBtn.style.color = '#dc3545');
    deleteBtn.onmouseout = () => (deleteBtn.style.color = '#6c757d');
    deleteBtn.onclick = e => {
        e.stopPropagation();
        layerDiv.remove();
        state.scribbles = state.scribbles.filter(stroke => stroke.layerId !== layerId);
        state.visibleLayerIds = state.visibleLayerIds.filter(id => id !== layerId);
        redrawAnnotations();
    };

    layerControls.append(visibilityToggle, colorPickerContainer, deleteBtn);
    layerDiv.append(layerName, layerControls);
    container.appendChild(layerDiv);

    if (!state.visibleLayerIds.includes(layerId)) {
        state.visibleLayerIds.push(layerId);
    }

    return layerId;
}

export function toggleLayerVisibility(layerId) {
    const layerEl = document.getElementById(layerId);
    const isVisible = layerEl.querySelector('input[type="checkbox"]').checked;
    if (isVisible) {
        if (!state.visibleLayerIds.includes(layerId)) {
            state.visibleLayerIds.push(layerId);
        }
    } else {
        state.visibleLayerIds = state.visibleLayerIds.filter(id => id !== layerId);
    }
    redrawAnnotations();
}
