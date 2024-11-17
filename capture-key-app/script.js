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
  generateSequences();
  displayCurrentSequence();
}

async function synchronizeTime() {
  try {
    const response = await fetch(`${SERVER_URL}/server_time`);
    const serverTime = await response.json();
    const clientTime = performance.now();
    timeOffset = serverTime.timestamp - clientTime;
    console.log(`Time offset calculated: ${timeOffset} ms`);
  } catch (error) {
    console.error("Error synchronizing time:", error);
    alert("Unable to synchronize time with the server.");
  }
}

function generateSequences() {
  sequences = [];

  for (let group = 0; group < totalRounds; group++) {
    let digitPool = [];
    for (let i = 0; i < 10; i++) {
      for (let d = 0; d < 10; d++) {
        digitPool.push(String(d));
      }
    }

    digitPool = shuffleArray(digitPool);

    let groupDigits = [];
    while (digitPool.length > 0) {
      let digit = digitPool.pop();

      if (
        groupDigits.length > 0 &&
        digit === groupDigits[groupDigits.length - 1]
      ) {
        let swapIndex = findNonRepeatingDigitIndex(digitPool, digit);
        if (swapIndex !== -1) {
          [digitPool[swapIndex], digit] = [digit, digitPool[swapIndex]];
          groupDigits.push(digit);
        } else {
          groupDigits.push(digit);
        }
      } else {
        groupDigits.push(digit);
      }
    }

    for (let i = 0; i < sequencesPerGroup; i++) {
      let sequence = groupDigits.slice(
        i * sequenceLength,
        (i + 1) * sequenceLength
      );
      sequences.push(sequence);
    }
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

  const rawTimestamp = performance.now();
  const timestamp = rawTimestamp + timeOffset;
  const key = event.key;
  const eventType = event.type;

  if (digits.includes(key) || key === "Backspace" || key === "Enter") {
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
      currentInputPosition++;
    }

    if (currentInputPosition < 0) {
      currentInputPosition = 0;
    }

    if (key === "Enter" && currentInputPosition >= sequenceLength) {
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
});
