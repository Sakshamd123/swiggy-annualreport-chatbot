/**
 * Swiggy Annual Report AI Assistant
 * Connects to POST /api/v1/query on the FastAPI backend.
 */

const API_ENDPOINT = '/api/v1/query';

// ─── DOM refs ───────────────────────────────────────────
const questionInput = document.getElementById('question-input');
const askBtn = document.getElementById('ask-btn');
const btnText = document.getElementById('btn-text');
const btnSpinner = document.getElementById('btn-spinner');
const errorBanner = document.getElementById('error-banner');
const errorMsg = document.getElementById('error-msg');
const answerSection = document.getElementById('answer-section');
const answerText = document.getElementById('answer-text');
const confidenceVal = document.getElementById('confidence-val');
const pagesVal = document.getElementById('pages-val');
const categoryVal = document.getElementById('category-val');
const contextSection = document.getElementById('context-section');
const contextText = document.getElementById('context-text');

// ─── State ───────────────────────────────────────────────
let isLoading = false;

// ─── Entry point ─────────────────────────────────────────
async function handleQuery() {
    const question = questionInput.value.trim();

    if (!question) {
        showError('Please enter a question before clicking Ask.');
        questionInput.focus();
        return;
    }

    if (isLoading) return;

    setLoading(true);
    hideError();
    hideResults();

    try {
        const data = await postQuery(question);
        renderResults(data);
    } catch (err) {
        showError(err.message || 'Something went wrong. Please try again.');
    } finally {
        setLoading(false);
    }
}

// ─── API call ────────────────────────────────────────────
async function postQuery(question) {
    let response;

    try {
        response = await fetch(API_ENDPOINT, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question }),
        });
    } catch (_) {
        throw new Error(
            'Could not reach the server. Make sure the backend is running.'
        );
    }

    if (!response.ok) {
        let detail = `Server returned ${response.status}`;
        try {
            const json = await response.json();
            if (json.detail) detail = json.detail;
        } catch (_) { }
        throw new Error(detail);
    }

    const json = await response.json();

    // Validate required fields
    if (!json.answer) throw new Error('Unexpected response format from server.');

    return json;
}

// ─── Render ──────────────────────────────────────────────
function renderResults(data) {
    // Answer
    answerText.textContent = data.answer;

    // Confidence (with colour class)
    const conf = (data.confidence || '').toUpperCase();
    confidenceVal.textContent = conf;
    confidenceVal.className = `meta-value confidence-${conf}`;

    // Source pages
    pagesVal.textContent = data.source_pages || '—';

    // Category
    categoryVal.textContent = formatCategory(data.category);

    // Show answer card
    answerSection.hidden = false;

    // Context snippet (optional)
    if (data.context_snippet && data.context_snippet.trim()) {
        contextText.textContent = data.context_snippet.trim();
        contextSection.hidden = false;
    } else {
        contextSection.hidden = true;
    }
}

// ─── UI helpers ──────────────────────────────────────────
function setLoading(loading) {
    isLoading = loading;
    askBtn.disabled = loading;
    questionInput.disabled = loading;
    btnText.textContent = loading ? 'Asking…' : 'Ask';
    btnSpinner.hidden = !loading;
}

function showError(msg) {
    errorMsg.textContent = msg;
    errorBanner.hidden = false;
}

function hideError() {
    errorBanner.hidden = true;
    errorMsg.textContent = '';
}

function hideResults() {
    answerSection.hidden = true;
    contextSection.hidden = true;
}

function formatCategory(raw) {
    if (!raw) return '—';
    // "FINANCIAL" → "Financial" etc.
    return raw.charAt(0).toUpperCase() + raw.slice(1).toLowerCase();
}

// ─── Keyboard submit (Enter key) ─────────────────────────
questionInput.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleQuery();
    }
});
