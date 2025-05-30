// website/frontend/static/js/color-picker.js

import { state } from './state.js';

export function initColorPicker() {
    const colorPicker = document.getElementById("colorPicker");

    if (!colorPicker) return;

    colorPicker.value = state.selectedColor;

    // Style
    Object.assign(colorPicker.style, {
        cursor: 'pointer',
        width: '38px',
        height: '38px',
        padding: '0',
        border: 'none',
        borderRadius: '8px'
    });

    colorPicker.addEventListener("input", e => {
        state.selectedColor = e.target.value;
    });
}