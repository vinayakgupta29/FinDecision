const $ = (selector) => document.querySelector(selector);

const form = $("#analysis-form");
const tabs = document.querySelectorAll(".tab");

const textGroup = $("#text-input-group");
const urlGroup = $("#url-input-group");

const textInput = $("#financial-text");
const urlInput = $("#article-url");
const charCount = $("#char-count");

const analyzeButton = $("#analyze-button");
const buttonLabel = $("#button-label");

const emptyState = $("#empty-state");
const loadingState = $("#loading-state");
const errorState = $("#error-state");
const resultContent = $("#result-content");

const resultStatus = $("#result-status");
const errorMessage = $("#error-message");
const loadingMessage = $("#loading-message");

const predictedClass = $("#predicted-class");
const predictionDescription = $("#prediction-description");
const probabilityList = $("#probability-list");

const articleMetadata = $("#article-metadata");
const articleTitle = $("#article-title");
const articleLink = $("#article-link");

const warningsContainer = $("#warnings-container");
const warningsList = $("#warnings-list");
const modelName = $("#model-name");

let inputMode = "text";

function setMode(mode) {
    inputMode = mode;

    for (const tab of tabs) {
        const active = tab.dataset.mode === mode;

        tab.classList.toggle("active", active);
        tab.setAttribute("aria-selected", String(active));
    }

    textGroup.hidden = mode !== "text";
    urlGroup.hidden = mode !== "url";

    buttonLabel.textContent =
        mode === "text" ? "Analyze sentiment" : "Analyze article";

    clearResult();
}

function clearResult() {
    emptyState.hidden = false;
    loadingState.hidden = true;
    errorState.hidden = true;
    resultContent.hidden = true;
    resultStatus.textContent = "Waiting for input";
}

function showLoading() {
    emptyState.hidden = true;
    loadingState.hidden = false;
    errorState.hidden = true;
    resultContent.hidden = true;
    resultStatus.textContent = "Analyzing";
    loadingMessage.textContent =
        inputMode === "url"
            ? "Fetching and analyzing article…"
            : "Running model inference…";
    analyzeButton.disabled = true;
}

function showError(message) {
    emptyState.hidden = true;
    loadingState.hidden = true;
    errorState.hidden = false;
    resultContent.hidden = true;
    resultStatus.textContent = "Failed";
    errorMessage.textContent = message;
}

function labelStyle(label) {
    const normalized = label.toLowerCase();

    if (normalized.includes("positive")) return "positive";
    if (normalized.includes("negative")) return "negative";
    if (normalized.includes("neutral")) return "neutral";

    return "other";
}

function formatLabel(label) {
    return label
        .replaceAll("_", " ")
        .replace(/\b\w/g, (character) => character.toUpperCase());
}

function renderProbabilities(probabilities) {
    probabilityList.replaceChildren();

    const entries = Object.entries(probabilities)
        .sort((a, b) => b[1] - a[1]);

    for (const [label, probability] of entries) {
        const row = document.createElement("div");
        row.className = "probability-row";

        const labelLine = document.createElement("div");
        labelLine.className = "probability-label";

        const name = document.createElement("span");
        name.textContent = formatLabel(label);

        const value = document.createElement("span");
        value.textContent = `${(probability * 100).toFixed(2)}%`;

        labelLine.append(name, value);

        const track = document.createElement("div");
        track.className = "track";

        const bar = document.createElement("div");
        bar.className = `bar ${labelStyle(label)}`;

        // Clamp display width against floating-point anomalies.
        const width = Math.max(
            0,
            Math.min(100, probability * 100)
        );

        bar.style.width = `${width}%`;
        track.append(bar);

        row.append(labelLine, track);
        probabilityList.append(row);
    }
}

function renderWarnings(warnings) {
    warningsList.replaceChildren();

    if (!warnings || warnings.length === 0) {
        warningsContainer.hidden = true;
        return;
    }

    for (const warning of warnings) {
        const item = document.createElement("li");
        item.textContent = warning;
        warningsList.append(item);
    }

    warningsContainer.hidden = false;
}

function renderResult(data) {
    emptyState.hidden = true;
    loadingState.hidden = true;
    errorState.hidden = true;
    resultContent.hidden = false;

    resultStatus.textContent = "Complete";

    predictedClass.textContent = formatLabel(
        data.predicted_class
    );

    predictionDescription.textContent =
        "Highest-scoring sentiment class for the supplied text.";

    renderProbabilities(data.probabilities);
    renderWarnings(data.warnings);

    modelName.textContent = data.model || "Unknown";

    if (data.article) {
        articleMetadata.hidden = false;

        articleTitle.textContent =
            data.article.title || "Untitled article";

        // Use the API-provided URL only after parsing it as HTTP(S).
        try {
            const parsed = new URL(data.article.source_url);

            if (!["http:", "https:"].includes(parsed.protocol)) {
                throw new Error("Unsupported URL scheme");
            }

            articleLink.href = parsed.href;
            articleLink.textContent = parsed.href;
        } catch {
            articleLink.removeAttribute("href");
            articleLink.textContent = "Source URL unavailable";
        }
    } else {
        articleMetadata.hidden = true;
        articleLink.removeAttribute("href");
    }
}

function errorText(data) {
    const detail = data?.detail;

    if (typeof detail === "string") {
        return detail;
    }

    if (detail?.message) {
        return detail.message;
    }

    if (data?.error?.message) {
        return data.error.message;
    }

    if (Array.isArray(detail)) {
        return detail
            .map((item) => item.msg || "Invalid input")
            .join("; ");
    }

    return "The request failed. Check that the API and model are running.";
}

tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
        setMode(tab.dataset.mode);
    });
});

textInput.addEventListener("input", () => {
    charCount.textContent =
        `${textInput.value.length.toLocaleString()} / 50,000`;
});

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    let endpoint;
    let payload;

    if (inputMode === "text") {
        const text = textInput.value.trim();

        if (!text) {
            showError("Enter financial text before submitting.");
            return;
        }

        payload = { text };
        endpoint = "/predict/text";
    } else {
        const url = urlInput.value.trim();

        if (!url) {
            showError("Enter an article URL before submitting.");
            return;
        }

        try {
            const parsed = new URL(url);

            if (!["http:", "https:"].includes(parsed.protocol)) {
                throw new Error("Only HTTP(S) URLs are supported.");
            }
        } catch (error) {
            showError(error.message || "Enter a valid article URL.");
            return;
        }

        payload = { url };
        endpoint = "/predict/url";
    }

    showLoading();

    try {
        const response = await fetch(endpoint, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            body: JSON.stringify(payload),
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(errorText(data));
        }

        renderResult(data);
    } catch (error) {
        showError(
            error.message ||
            "Could not connect to FinDecision."
        );
    } finally {
        analyzeButton.disabled = false;
    }
});

async function checkHealth() {
    try {
        const response = await fetch("/health");
        const data = await response.json();

        if (!response.ok || !data.model_loaded) {
            resultStatus.textContent = "Model unavailable";
        }
    } catch {
        resultStatus.textContent = "API unavailable";
    }
}

checkHealth();
