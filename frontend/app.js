const input = document.querySelector("#imageInput");
const dropZone = document.querySelector("#dropZone");
const uploadPrompt = document.querySelector("#uploadPrompt");
const preview = document.querySelector("#preview");
const fileMeta = document.querySelector("#fileMeta");
const analyzeButton = document.querySelector("#analyzeButton");
const clearButton = document.querySelector("#clearButton");
const serviceStatus = document.querySelector("#serviceStatus");
const emptyState = document.querySelector("#emptyState");
const loadingState = document.querySelector("#loadingState");
const result = document.querySelector("#result");
const errorMessage = document.querySelector("#errorMessage");

const MAX_BYTES = 10 * 1024 * 1024;
const ALLOWED_TYPES = new Set(["image/jpeg", "image/png"]);
let selectedFile = null;
let previewUrl = null;

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function showError(message) {
  emptyState.hidden = true;
  loadingState.hidden = true;
  result.hidden = true;
  errorMessage.textContent = message;
  errorMessage.hidden = false;
}

function resetResult() {
  errorMessage.hidden = true;
  result.hidden = true;
  loadingState.hidden = true;
  emptyState.hidden = false;
}

function clearSelection() {
  selectedFile = null;
  input.value = "";
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  preview.hidden = true;
  preview.removeAttribute("src");
  uploadPrompt.hidden = false;
  fileMeta.textContent = "";
  analyzeButton.disabled = true;
  clearButton.hidden = true;
  resetResult();
}

function selectFile(file) {
  if (!file) return;
  if (!ALLOWED_TYPES.has(file.type)) {
    clearSelection();
    showError("Choose a JPEG or PNG image.");
    return;
  }
  if (file.size > MAX_BYTES) {
    clearSelection();
    showError("The image must be 10 MB or smaller.");
    return;
  }

  selectedFile = file;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  preview.src = previewUrl;
  preview.hidden = false;
  uploadPrompt.hidden = true;
  fileMeta.textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB`;
  analyzeButton.disabled = false;
  clearButton.hidden = false;
  resetResult();
}

async function analyze() {
  if (!selectedFile) return;
  analyzeButton.disabled = true;
  emptyState.hidden = true;
  errorMessage.hidden = true;
  result.hidden = true;
  loadingState.hidden = false;

  const form = new FormData();
  form.append("file", selectedFile);

  try {
    const response = await fetch("/api/predict", { method: "POST", body: form });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "The analysis could not be completed.");

    const flagged = payload.label === "PNEUMONIA";
    const banner = document.querySelector("#resultBanner");
    banner.classList.toggle("flagged", flagged);
    document.querySelector("#resultLabel").textContent = flagged
      ? "Pneumonia signal flagged"
      : "No pneumonia signal flagged";
    document.querySelector("#resultConfidence").textContent = `${formatPercent(payload.confidence)} model confidence`;
    document.querySelector("#normalValue").textContent = formatPercent(payload.normal_probability);
    document.querySelector("#pneumoniaValue").textContent = formatPercent(payload.pneumonia_probability);
    document.querySelector("#normalBar").style.width = formatPercent(payload.normal_probability);
    document.querySelector("#pneumoniaBar").style.width = formatPercent(payload.pneumonia_probability);
    document.querySelector("#disclaimer").textContent = payload.disclaimer;
    loadingState.hidden = true;
    result.hidden = false;
  } catch (error) {
    showError(error.message);
  } finally {
    analyzeButton.disabled = false;
  }
}

async function loadStatus() {
  try {
    const response = await fetch("/api/health");
    const payload = await response.json();
    const ready = response.ok && payload.model_available;
    serviceStatus.textContent = ready ? "Model ready" : "Model unavailable";
    serviceStatus.className = `service-status ${ready ? "online" : "offline"}`;
  } catch {
    serviceStatus.textContent = "Service unavailable";
    serviceStatus.className = "service-status offline";
  }
}

async function loadMetrics() {
  try {
    const response = await fetch("/api/metrics");
    const payload = await response.json();
    if (!payload.available) return;
    const values = [
      ["ROC-AUC", payload.roc_auc.toFixed(3)],
      ["Accuracy", formatPercent(payload.accuracy)],
      ["Sensitivity", formatPercent(payload.pneumonia_recall)],
      ["Specificity", payload.specificity == null ? "N/A" : formatPercent(payload.specificity)],
    ];
    document.querySelector("#metrics").innerHTML = values
      .map(([label, value]) => `<div class="metric"><span>${label}</span><strong>${value}</strong></div>`)
      .join("");
  } catch {
    // The page remains usable when metrics are absent.
  }
}

input.addEventListener("change", () => selectFile(input.files[0]));
clearButton.addEventListener("click", clearSelection);
analyzeButton.addEventListener("click", analyze);
["dragenter", "dragover"].forEach((eventName) =>
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("dragging");
  }),
);
["dragleave", "drop"].forEach((eventName) =>
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragging");
  }),
);
dropZone.addEventListener("drop", (event) => selectFile(event.dataTransfer.files[0]));

loadStatus();
loadMetrics();
