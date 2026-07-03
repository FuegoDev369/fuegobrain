/**
 * app.js
 * Logique de la démo web FuegoBrain — vanilla JS, zéro dépendance externe.
 * Fait un seul appel POST /orchestrate (bloquant côté serveur) et anime
 * le pipeline côté client avec des timings estimés pendant que la requête
 * est en vol. Les vraies durées par agent sont affichées APRÈS réception
 * de la réponse, depuis pipeline_trace — voir DEC-05 / DEC-10.
 */

// ── Constantes ──────────────────────────────────────────────────────────

const API_BASE = window.location.origin; // auto-détecte localhost vs Render
const AGENT_NAMES = ['researcher', 'reasoner', 'synthesizer'];

// Durées estimées pour l'animation progressive (ms).
// Purement cosmétique — la vraie durée de chaque agent est affichée
// après coup, depuis pipeline_trace.
const STAGE_ANIMATION_DURATIONS = {
  researcher: 3000,
  reasoner: 3000,
  synthesizer: 2000,
};

// Drapeau global qui permet de couper l'animation dès que la vraie
// réponse API arrive (ou en cas d'erreur), sans laisser de timers
// continuer à tourner en arrière-plan.
let animationCancelled = false;

// ── Initialisation ───────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', initEventListeners);

/**
 * Attache tous les event listeners de la page.
 */
function initEventListeners() {
  const submitBtn = document.getElementById('submit-btn');
  const queryInput = document.getElementById('query-input');
  const exampleBtns = document.querySelectorAll('.example-btn');
  const toggleRawBtn = document.getElementById('toggle-raw-btn');

  submitBtn.addEventListener('click', validateAndSubmit);

  // Entrée envoie la requête, Shift+Entrée fait un retour à la ligne
  queryInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      validateAndSubmit();
    }
  });

  exampleBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      queryInput.value = btn.dataset.query;
      queryInput.focus();
    });
  });

  if (toggleRawBtn) {
    toggleRawBtn.addEventListener('click', () => {
      const rawJson = document.getElementById('raw-json');
      const traceContainer = document.getElementById('trace-container');
      const showingRaw = !rawJson.classList.contains('hidden');

      if (showingRaw) {
        rawJson.classList.add('hidden');
        traceContainer.classList.remove('hidden');
        toggleRawBtn.textContent = 'View Raw JSON';
      } else {
        rawJson.classList.remove('hidden');
        traceContainer.classList.add('hidden');
        toggleRawBtn.textContent = 'View Trace';
      }
    });
  }
}

// ── Soumission de la requête ─────────────────────────────────────────────

/**
 * Valide l'input utilisateur puis lance le pipeline : animation cosmétique
 * en parallèle du vrai appel réseau vers /orchestrate.
 */
async function validateAndSubmit() {
  const queryInput = document.getElementById('query-input');
  const submitBtn = document.getElementById('submit-btn');
  const query = queryInput.value.trim();

  if (query.length < 10) {
    showInlineError('Please enter at least 10 characters.');
    return;
  }

  // Reset UI avant de lancer un nouveau run
  clearInlineError();
  submitBtn.disabled = true;
  submitBtn.textContent = 'Running…';
  document.getElementById('pipeline-section').classList.remove('hidden');
  document.getElementById('results-section').classList.add('hidden');
  resetStages();

  animationCancelled = false;
  // L'animation tourne sans être attendue — purement cosmétique,
  // elle ne doit jamais bloquer le vrai appel réseau.
  startPipelineAnimation();

  try {
    const data = await callOrchestrate(query);
    stopPipelineAnimation();
    displayResults(data);
  } catch (error) {
    stopPipelineAnimation();
    showInlineError(error.message || 'Something went wrong. Please try again.');
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = 'Run Pipeline →';
  }
}

/**
 * Appelle l'API FuegoBrain — le seul vrai appel réseau de la démo.
 */
async function callOrchestrate(query) {
  const response = await fetch(`${API_BASE}/orchestrate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

// ── Animation cosmétique du pipeline ──────────────────────────────────────

/**
 * Anime les 3 stages en séquence avec des timings estimés, pendant que
 * le vrai appel /orchestrate est en vol. S'arrête immédiatement si
 * stopPipelineAnimation() a déjà été appelée (réponse API arrivée tôt).
 */
async function startPipelineAnimation() {
  for (const agent of AGENT_NAMES) {
    if (animationCancelled) return;
    setStageState(agent, 'pending');
    await sleep(STAGE_ANIMATION_DURATIONS[agent]);
    if (animationCancelled) return;
    setStageState(agent, 'complete');
  }
}

/**
 * Coupe l'animation et force tous les stages restants en "complete"
 * instantanément — appelé dès que la vraie réponse API arrive.
 */
function stopPipelineAnimation() {
  animationCancelled = true;
  AGENT_NAMES.forEach((agent) => setStageState(agent, 'complete'));
}

/**
 * Met à jour les classes CSS d'une carte de stage ('idle' | 'pending' | 'complete').
 */
function setStageState(agent, state) {
  const el = document.getElementById(`stage-${agent}`);
  if (!el) return;
  el.classList.remove('idle', 'pending', 'complete');
  el.classList.add(state);
}

function resetStages() {
  AGENT_NAMES.forEach((agent) => setStageState(agent, 'idle'));
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ── Affichage des résultats ────────────────────────────────────────────

/**
 * Affiche la réponse finale, les métadonnées, et la trace détaillée
 * du pipeline une fois la vraie réponse API reçue.
 */
function displayResults(data) {
  document.getElementById('results-section').classList.remove('hidden');

  // Réponse finale, formatée en mini-Markdown
  document.getElementById('final-answer').innerHTML = formatMarkdown(data.final_answer);

  // Barre de métadonnées agrégées
  // AVANT (TICKET-16 original) : data.metadata.model — champ retiré par TICKET-33
  // (PipelineMetadata n'a plus de valeur unique de modèle, structurellement
  // incapable de représenter 3 modèles potentiellement différents par agent).
  // APRÈS (TICKET-37) : data.metadata.models_used est une liste dédupliquée,
  // ordre d'exécution préservé — un seul élément dans la config par défaut
  // (les 3 agents partagent gemini/gemini-2.5-flash), potentiellement
  // plusieurs pour une config mixte. Jointe ici pour un affichage compact.
  const totalTokens = data.metadata.total_input_tokens + data.metadata.total_output_tokens;
  document.getElementById('metadata-bar').textContent =
    `Pipeline: ${data.metadata.total_duration_ms}ms · ${totalTokens} tokens · ${data.metadata.models_used.join(', ')}`;

  // Durées réelles par agent, injectées dans les cartes de stage
  data.pipeline_trace.forEach((item) => {
    const durationEl = document.getElementById(`duration-${item.agent}`);
    if (durationEl) durationEl.textContent = `${item.duration_ms}ms`;
  });

  // Trace détaillée (accordion simple) — prompt envoyé + réponse brute
  const traceContainer = document.getElementById('trace-container');
  traceContainer.innerHTML = '';
  data.pipeline_trace.forEach((item) => {
    const trace = document.createElement('div');
    trace.className = 'trace-item';

    const title = document.createElement('div');
    title.className = 'trace-item-title';
    title.textContent = `${item.agent} — ${item.duration_ms}ms · ${item.input_tokens}+${item.output_tokens} tokens`;

    const body = document.createElement('pre');
    body.textContent = `PROMPT SENT:\n${item.prompt_sent}\n\nRESPONSE:\n${item.response}`;

    trace.appendChild(title);
    trace.appendChild(body);
    traceContainer.appendChild(trace);
  });

  // JSON brut pour debug / transparence totale
  document.getElementById('raw-json').textContent = JSON.stringify(data, null, 2);
}

/**
 * Mini-parser Markdown vanilla, volontairement minimal (pas de lib externe).
 * Couvre seulement ce que produit le Synthesizer : ## headers, **bold**,
 * listes à puces, et paragraphes séparés par des lignes vides.
 */
function formatMarkdown(text) {
  const lines = text.split('\n');
  let html = '';
  let inList = false;

  for (const rawLine of lines) {
    const line = rawLine.trim();

    if (line.startsWith('## ')) {
      if (inList) { html += '</ul>'; inList = false; }
      html += `<h3>${line.slice(3)}</h3>`;
      continue;
    }

    if (line.startsWith('- ') || line.startsWith('* ')) {
      if (!inList) { html += '<ul>'; inList = true; }
      html += `<li>${inlineMarkdown(line.slice(2))}</li>`;
      continue;
    }

    if (inList) { html += '</ul>'; inList = false; }

    if (line === '') {
      html += '';
    } else {
      html += `<p>${inlineMarkdown(line)}</p>`;
    }
  }

  if (inList) html += '</ul>';
  return html;
}

/**
 * Applique le formatage inline (**bold**) sur une ligne de texte.
 */
function inlineMarkdown(line) {
  return line.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
}

// ── Erreurs inline ────────────────────────────────────────────────────

function showInlineError(message) {
  let errorEl = document.getElementById('inline-error');
  if (!errorEl) {
    errorEl = document.createElement('div');
    errorEl.id = 'inline-error';
    errorEl.className = 'inline-error';
    const inputSection = document.querySelector('.input-section');
    inputSection.appendChild(errorEl);
  }
  errorEl.textContent = message;
}

function clearInlineError() {
  const errorEl = document.getElementById('inline-error');
  if (errorEl) errorEl.remove();
}
