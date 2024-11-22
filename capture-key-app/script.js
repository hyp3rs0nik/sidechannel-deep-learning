// script.js

const digits = [...Array(10).keys()].map(String);
const groupSize = 100;
const sequenceLength = 5;
const sequencesPerGroup = groupSize / sequenceLength;
const debounceDelay = 200;
const SERVER_URL = "http://192.168.1.139:3000";

let totalRounds = 0;
let sequences = [];
let currentSequenceIndex = 0;
let typedData = [];
let isTaskRunning = false;
let currentInputPosition = 0;
let timeOffset = 0;

const roundButtons = document.querySelectorAll(".round-button");
const sequenceDisplay = document.getElementById("sequence-display");
const typingArea = document.getElementById("typing-area");
const progressBar = document.getElementById("progress-bar");
const statusDiv = document.getElementById("status");

const syncIntervalMs = 5 * 60 * 1000; // Every 5 minutes

roundButtons.forEach((button) => {
  button.addEventListener("click", () => {
    if (isTaskRunning) return;
    const rounds = parseInt(button.getAttribute("data-rounds"), 10);
    selectRound(rounds, button);
  });
});

typingArea.addEventListener("keydown", recordKeystroke);

async function selectRound(rounds, button) {
  totalRounds = rounds;
  isTaskRunning = true;
  typedData = [];
  currentSequenceIndex = 0;
  statusDiv.textContent = "Task in progress...";
  toggleRoundButtons(true);
  button.classList.add("selected");

  await synchronizeTime();
  startSynchronization();
  generateSequences();
  displayCurrentSequence();
}

async function synchronizeTime() {
  try {
    const t0 = Date.now();
    const response = await fetch(`${SERVER_URL}/server_time`);
    const t3 = Date.now();
    const serverTime = await response.json();
    const t1 = serverTime.timestamp;
    const rtt = t3 - t0;
    timeOffset = t1 - t3 + rtt / 2;
    console.log(`Time offset calculated: ${timeOffset} ms`);
  } catch (error) {
    console.error("Error synchronizing time:", error);
    alert("Unable to synchronize time with the server.");
  }
}

function startSynchronization() {
  setInterval(synchronizeTime, syncIntervalMs);
}

function generateSequences() {
  sequences = [];

  const totalSequences = totalRounds * sequencesPerGroup;
  const totalDigits = totalSequences * sequenceLength;
  const digitsPerDigit = totalDigits / 10;

  if (totalDigits % 10 !== 0) {
    alert("Total number of digits is not a multiple of 10. Cannot ensure equal distribution.");
    return;
  }

  let digitPool = [];
  for (let i = 0; i < 10; i++) {
    for (let j = 0; j < digitsPerDigit; j++) {
      digitPool.push(String(i));
    }
  }

  digitPool = shuffleArray(digitPool);

  let allDigits = [];
  while (digitPool.length > 0) {
    let digit = digitPool.pop();

    if (
      allDigits.length > 0 &&
      digit === allDigits[allDigits.length - 1]
    ) {
      let swapIndex = findNonRepeatingDigitIndex(digitPool, digit);
      if (swapIndex !== -1) {
        [digitPool[swapIndex], digit] = [digit, digitPool[swapIndex]];
        allDigits.push(digit);
      } else {
        allDigits.push(digit);
      }
    } else {
      allDigits.push(digit); 
    }
  }

  for (let i = 0; i < allDigits.length; i += sequenceLength) {
    let sequence = allDigits.slice(i, i + sequenceLength);
    sequences.push(sequence);
  }
}


function findNonRepeatingDigitIndex(digitPool, currentDigit) {
  for (let i = digitPool.length - 1; i >= 0; i--) {
    if (digitPool[i] !== currentDigit) {
      return i;
    }
  }
  return -1;
}

function shuffleArray(array) {
  let m = array.length,
    t,
    i;
  while (m) {
    i = Math.floor(Math.random() * m--);
    t = array[m];
    array[m] = array[i];
    array[i] = t;
  }
  return array;
}

function displayCurrentSequence() {
  if (currentSequenceIndex < sequences.length) {
    let sequence = sequences[currentSequenceIndex];
    sequenceDisplay.textContent = sequence.join(" ");
    typingArea.value = "";
    typingArea.disabled = false;
    typingArea.focus();
    currentInputPosition = 0;
    updateProgressBar();
    statusDiv.textContent = `Sequence ${currentSequenceIndex + 1} of ${
      sequences.length
    }`;
  } else {
    endTask();
  }
}

function updateProgressBar() {
  const progressPercent = (currentSequenceIndex / sequences.length) * 100;
  progressBar.style.width = `${progressPercent}%`;
}

function recordKeystroke(event) {
  if (!isTaskRunning) return;

  const rawTimestamp = Date.now();
  const timestamp = rawTimestamp + timeOffset;
  const key = event.key;
  const eventType = event.type;

  // Allow only digits and Backspace
  if (digits.includes(key) || key === "Backspace") {
    typedData.push({
      timestamp,
      key,
      sequenceIndex: currentSequenceIndex,
    });
  } else {
    event.preventDefault();
    return;
  }

  if (eventType === "keydown") {
    if (key === "Backspace") {
      if (currentInputPosition > 0) {
        currentInputPosition--;
      }
    } else if (digits.includes(key)) {
      if (currentInputPosition < sequenceLength) {
        currentInputPosition++;
      } else {
        // Prevent further input if sequence length is reached
        event.preventDefault();
        return;
      }
    }

    if (currentInputPosition === sequenceLength) {
      typingArea.disabled = true;
      setTimeout(() => {
        currentSequenceIndex++;
        displayCurrentSequence();
      }, debounceDelay);
    }
  }
}
function endTask() {
  isTaskRunning = false;
  typingArea.disabled = true;
  typingArea.value = "";
  toggleRoundButtons(false);
  sequenceDisplay.textContent = "Task completed!";
  statusDiv.textContent = "Thank you for completing the task.";
  progressBar.style.width = "100%";
  sendKeystrokeDataToServer();
}

async function sendKeystrokeDataToServer() {
  try {
    const response = await fetch(`${SERVER_URL}/keystroke_data`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(typedData),
    });
    const result = await response.json();
    console.log("Keystroke data sent successfully:", result);
  } catch (error) {
    console.error("Error sending keystroke data:", error);
    alert("Failed to send keystroke data to the server.");
  }
}

function toggleRoundButtons(disabled) {
  roundButtons.forEach((button) => {
    button.disabled = disabled;
    if (!disabled) {
      button.classList.remove("selected");
    }
  });
}

document.addEventListener("DOMContentLoaded", () => {
  statusDiv.textContent = "Select a number of rounds to begin.";

  typingArea.setAttribute("autocomplete", "off");
  typingArea.setAttribute("autocorrect", "off");
  typingArea.setAttribute("autocapitalize", "off");
  typingArea.setAttribute("spellcheck", "false");

  synchronizeTime(); // Initial synchronization
  startSynchronization(); // Periodic resynchronization
});
