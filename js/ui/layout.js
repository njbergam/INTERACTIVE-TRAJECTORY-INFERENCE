import { elements } from '../dom.js';
import { resetAppState } from '../state.js';

export function ensureResizeControl(updateVisualization) {
    if (document.getElementById('resize-plots-control')) return;

    const resizeContainer = document.createElement('div');
    resizeContainer.id = 'resize-plots-control';
    resizeContainer.style.marginTop = '10px';

    const resizeLabel = document.createElement('label');
    resizeLabel.textContent = 'Resize Plots: ';

    const resizeSlider = document.createElement('input');
    resizeSlider.type = 'range';
    resizeSlider.min = '200';
    resizeSlider.max = '800';
    resizeSlider.value = '400';
    resizeSlider.addEventListener('input', () => updateVisualization());

    resizeContainer.appendChild(resizeLabel);
    resizeContainer.appendChild(resizeSlider);
    elements.vizContainer.prepend(resizeContainer);
}

export function getResizePlotSize(defaultSize) {
    const slider = document.querySelector('#resize-plots-control input[type=\"range\"]');
    const val = slider ? parseInt(slider.value, 10) : defaultSize;
    return Number.isFinite(val) ? val : defaultSize;
}

export function showHomePage(dataPickerVisible) {
    resetAppState();
    elements.elbowCanvases = [];

    elements.controls.style.display = 'none';
    elements.vizContainer.style.display = 'none';
    elements.dropZone.style.display = 'block';
    elements.dataPicker.style.display = dataPickerVisible ? 'block' : 'none';

    elements.status.textContent = 'Waiting for file...';
    elements.tooltip.style.display = 'none';
    if (elements.transportNote) elements.transportNote.style.display = 'none';

    elements.epsSlider.value = '0';
    elements.epsLabel.textContent = 'Regularization (ε): Exact ';

    elements.kContainer.innerHTML = '';
    elements.scatterContainer.innerHTML = '';
    document.getElementById('resize-plots-control')?.remove();
    if (elements.ctx && elements.canvas) {
        elements.ctx.clearRect(0, 0, elements.canvas.width, elements.canvas.height);
    }
    if (elements.globalInertiaCtx && elements.globalInertiaCanvas) {
        elements.globalInertiaCtx.clearRect(0, 0, elements.globalInertiaCanvas.width, elements.globalInertiaCanvas.height);
    }
    if (elements.globalTruePiCtx && elements.globalTruePiCanvas) {
        elements.globalTruePiCtx.clearRect(0, 0, elements.globalTruePiCanvas.width, elements.globalTruePiCanvas.height);
    }
}

