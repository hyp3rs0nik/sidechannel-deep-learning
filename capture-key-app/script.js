// script.js

// Configuration Variables
const digits = [...Array(10).keys()].map(String); // ["0", "1", ..., "9"]
const groupSize = 100; // Each group has 100 digits (10 digits * 10 times each)
const sequenceLength = 5; // Each sequence is 5 numbers
const sequencesPerGroup = groupSize / sequenceLength; // 100 / 5 = 20 sequences per group
const debounceDelay = 200; // milliseconds

// State Variables
let totalRounds = 0; // Number of groups selected
let sequences = []; // Array of sequences across all groups
let currentSequenceIndex = 0; // Index into sequences array
let typedData = []; // Records user's keystrokes and timestamps
let isTaskRunning = false;
let currentInputPosition = 0; // Position within the current sequence

// DOM Elements
const roundButtons = document.querySelectorAll('.round-button');
const sequenceDisplay = document.getElementById('sequence-display');
const typingArea = document.getElementById('typing-area');
const progressBar = document.getElementById('progress-bar');
const downloadDataButton = document.getElementById('download-data');
const clearDataButton = document.getElementById('clear-data');
const statusDiv = document.getElementById('status');

// Event Listeners for Round Buttons
roundButtons.forEach(button => {
    button.addEventListener('click', () => {
        if (isTaskRunning) return; // Prevent changing rounds mid-task
        const rounds = parseInt(button.getAttribute('data-rounds'), 10);
        selectRound(rounds, button);
    });
});

// Event Listeners for Action Buttons
downloadDataButton.addEventListener('click', downloadData);
clearDataButton.addEventListener('click', clearData);

// Event Listener for Typing Area
typingArea.addEventListener('keydown', recordKeystroke);

// Function to Handle Round Selection
function selectRound(rounds, button) {
    totalRounds = rounds;
    isTaskRunning = true;
    typedData = [];
    currentSequenceIndex = 0;
    statusDiv.textContent = 'Task in progress...';
    toggleRoundButtons(true);
    button.classList.add('selected');
    generateSequences();
    displayCurrentSequence();
}

// Function to Generate Sequences
function generateSequences() {
    sequences = []; // Reset sequences array

    for (let group = 0; group < totalRounds; group++) {
        // Create a pool of digits with 10 of each digit (0-9)
        let digitPool = [];
        for (let i = 0; i < 10; i++) {
            for (let d = 0; d < 10; d++) {
                digitPool.push(String(d));
            }
        }
        // Shuffle the digit pool
        digitPool = shuffleArray(digitPool);

        let groupDigits = [];
        while (digitPool.length > 0) {
            let digit = digitPool.pop();
            // Ensure no consecutive repeats
            if (groupDigits.length > 0 && digit === groupDigits[groupDigits.length - 1]) {
                // Find a non-repeating digit in the pool
                let swapIndex = findNonRepeatingDigitIndex(digitPool, digit);
                if (swapIndex !== -1) {
                    // Swap and push the non-repeating digit
                    [digitPool[swapIndex], digit] = [digit, digitPool[swapIndex]];
                    groupDigits.push(digit);
                } else {
                    // Accept repeat if no alternative
                    groupDigits.push(digit);
                }
            } else {
                groupDigits.push(digit);
            }
        }
        // Split groupDigits into sequences of 5 numbers
        for (let i = 0; i < sequencesPerGroup; i++) {
            let sequence = groupDigits.slice(i * sequenceLength, (i + 1) * sequenceLength);
            sequences.push(sequence);
        }
    }
}

// Utility Function to Find Index of a Non-Repeating Digit
function findNonRepeatingDigitIndex(digitPool, currentDigit) {
    for (let i = digitPool.length - 1; i >= 0; i--) {
        if (digitPool[i] !== currentDigit) {
            return i;
        }
    }
    return -1; // No non-repeating digit found
}

// Utility Function to Shuffle an Array (Fisher-Yates Shuffle)
function shuffleArray(array) {
    let m = array.length, t, i;
    while (m) {
        i = Math.floor(Math.random() * m--);
        t = array[m];
        array[m] = array[i];
        array[i] = t;
    }
    return array;
}

// Function to Display the Current Sequence
function displayCurrentSequence() {
    if (currentSequenceIndex < sequences.length) {
        let sequence = sequences[currentSequenceIndex];
        sequenceDisplay.textContent = sequence.join(' ');
        typingArea.value = '';
        typingArea.disabled = false;
        typingArea.focus();
        currentInputPosition = 0;
        updateProgressBar();
        statusDiv.textContent = `Sequence ${currentSequenceIndex + 1} of ${sequences.length}`;
    } else {
        endTask();
    }
}

// Function to Update Progress Bar
function updateProgressBar() {
    const progressPercent = (currentSequenceIndex / sequences.length) * 100;
    progressBar.style.width = `${progressPercent}%`;
}

// Function to Record Keystrokes
function recordKeystroke(event) {
    if (!isTaskRunning) return;

    const timestamp = Date.now();
    let actualKey = event.key;

    // Allow only digits and Backspace
    if (digits.includes(actualKey) || actualKey === 'Backspace') {
        typedData.push({ actualKey, timestamp });
    } else {
        event.preventDefault();
        return;
    }

    if (actualKey === 'Backspace') {
        if (currentInputPosition > 0) {
            currentInputPosition--;
        }
    } else if (digits.includes(actualKey)) {
        currentInputPosition++;
    }

    if (currentInputPosition < 0) {
        currentInputPosition = 0;
    }

    // Move to next sequence after typing 5 digits
    if (currentInputPosition >= sequenceLength) {
        typingArea.disabled = true;
        setTimeout(() => {
            currentSequenceIndex++;
            displayCurrentSequence();
        }, debounceDelay);
    }
}

// Function to End the Typing Task
function endTask() {
    isTaskRunning = false;
    typingArea.disabled = true;
    typingArea.value = '';
    toggleRoundButtons(false);
    sequenceDisplay.textContent = 'Task completed!';
    statusDiv.textContent = 'You can start a new session or download your data.';
    progressBar.style.width = '100%';
    downloadDataButton.disabled = typedData.length === 0;
    clearDataButton.disabled = typedData.length === 0;
}

// Function to Download Typed Data as CSV
function downloadData() {
    if (typedData.length === 0) return;

    let csvContent = "data:text/csv;charset=utf-8,actualKey,timestamp\n";
    typedData.forEach(function(row) {
        csvContent += `${row.actualKey},${row.timestamp}\n`;
    });

    const encodedUri = encodeURI(csvContent);
    const downloadAnchorNode = document.createElement('a');
    downloadAnchorNode.setAttribute("href", encodedUri);
    downloadAnchorNode.setAttribute("download", "typed_data.csv");
    document.body.appendChild(downloadAnchorNode);
    downloadAnchorNode.click();
    downloadAnchorNode.remove();

    statusDiv.textContent = 'Data downloaded! You can continue typing or clear the data.';
}

// Function to Clear Typed Data
function clearData() {
    typedData = [];
    clearDataButton.disabled = true;
    downloadDataButton.disabled = true;
    statusDiv.textContent = 'Data cleared! Start a new session whenever you are ready.';
}

// Function to Enable/Disable Round Buttons During Task
function toggleRoundButtons(disabled) {
    roundButtons.forEach(button => {
        button.disabled = disabled;
        if (!disabled) {
            button.classList.remove('selected');
        }
    });
    downloadDataButton.disabled = disabled || typedData.length === 0;
    clearDataButton.disabled = disabled || typedData.length === 0;
}

// Initial Setup
document.addEventListener('DOMContentLoaded', () => {
    statusDiv.textContent = 'Select a number of rounds to begin.';
});
