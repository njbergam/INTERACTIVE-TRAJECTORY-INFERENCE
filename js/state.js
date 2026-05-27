import { createDefaultState } from './constants.js';

export let rawSequences = [];
export let state = createDefaultState();
export let drawnEdgePaths = [];
export let drawnGlobalPoints = [];
export let dataPickerAvailable = false;

export function resetAppState() {
    rawSequences = [];
    state = createDefaultState();
    drawnEdgePaths = [];
    drawnGlobalPoints = [];
}

export function setRawSequences(seqs) {
    rawSequences = seqs;
}

export function setDataPickerAvailable(value) {
    dataPickerAvailable = value;
}
