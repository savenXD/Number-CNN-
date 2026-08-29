/**
 * Numerix AI - Canvas & FastAPI Backend Predictor Engine
 */

// =========================================================================
// DEPLOYMENT CONFIGURATION:
// Live Render backend service URL
// =========================================================================
const RENDER_BACKEND_URL = 'https://number-cnn.onrender.com'; 

document.addEventListener('DOMContentLoaded', () => {
  const BACKEND_URL = RENDER_BACKEND_URL || (
    (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
      ? 'http://localhost:8000'
      : 'http://localhost:8000'
  );
  
  // DOM Elements
  const canvas = document.getElementById('digitCanvas');
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const canvasPrompt = document.getElementById('canvasPrompt');
  
  const loaderOverlay = document.getElementById('loaderOverlay');
  const undoBtn = document.getElementById('undoBtn');
  const penBtn = document.getElementById('penBtn');
  const predictBtn = document.getElementById('predictBtn');
  
  const digitTicker = document.getElementById('digitTicker');
  const topScoreText = document.getElementById('topScoreText');
  const backendStatusBadge = document.getElementById('backendStatusBadge');
  const probabilityList = document.getElementById('probabilityList');

  // Canvas State Variables
  let isDrawing = false;
  let hasPredicted = false;
  let undoStack = [];
  const MAX_UNDO = 15;
  let dpr = window.devicePixelRatio || 1;

  // Enforce touch action none on canvas element
  canvas.style.touchAction = 'none';
  canvas.style.webkitUserSelect = 'none';
  canvas.style.userSelect = 'none';

  // 1. High DPI Canvas Setup
  function initCanvas() {
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    
    // Fill canvas background with white
    ctx.fillStyle = '#FFFFFF';
    ctx.fillRect(0, 0, rect.width, rect.height);
    
    setDrawingStyles();
    saveCanvasState();
    updateUndoState();
  }

  function setDrawingStyles() {
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.lineWidth = 18;
    ctx.strokeStyle = '#000000';
  }

  function getCanvasCoords(e) {
    const rect = canvas.getBoundingClientRect();
    let clientX = e.clientX;
    let clientY = e.clientY;
    
    if (e.touches && e.touches.length > 0) {
      clientX = e.touches[0].clientX;
      clientY = e.touches[0].clientY;
    }
    
    return {
      x: clientX - rect.left,
      y: clientY - rect.top
    };
  }

  function saveCanvasState() {
    const rect = canvas.getBoundingClientRect();
    if (undoStack.length >= MAX_UNDO) {
      undoStack.shift();
    }
    undoStack.push(ctx.getImageData(0, 0, rect.width * dpr, rect.height * dpr));
    updateUndoState();
  }

  function updateUndoState() {
    if (undoBtn) {
      undoBtn.disabled = false;
      undoBtn.classList.remove('opacity-30');
    }
  }

  function hidePrompt() {
    if (canvasPrompt) {
      canvasPrompt.classList.add('opacity-0');
    }
  }

  function showPrompt() {
    if (canvasPrompt) {
      canvasPrompt.classList.remove('opacity-0');
    }
  }

  // 2. Drawing Mechanics
  let lastPos = { x: 0, y: 0 };

  function startDrawing(e) {
    if (e.cancelable) e.preventDefault();
    if (e.stopPropagation) e.stopPropagation();

    if (hasPredicted) {
      hasPredicted = false;
      const rect = canvas.getBoundingClientRect();
      ctx.fillStyle = '#FFFFFF';
      ctx.fillRect(0, 0, rect.width, rect.height);
      undoStack = [];
      saveCanvasState();
    }

    isDrawing = true;
    hidePrompt();
    lastPos = getCanvasCoords(e);

    ctx.beginPath();
    ctx.moveTo(lastPos.x, lastPos.y);
    setDrawingStyles();
  }

  function draw(e) {
    if (!isDrawing) return;
    if (e.cancelable) e.preventDefault();
    if (e.stopPropagation) e.stopPropagation();
    
    const currentPos = getCanvasCoords(e);
    
    const midPoint = {
      x: (lastPos.x + currentPos.x) / 2,
      y: (lastPos.y + currentPos.y) / 2
    };

    ctx.quadraticCurveTo(lastPos.x, lastPos.y, midPoint.x, midPoint.y);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(midPoint.x, midPoint.y);
    lastPos = currentPos;
  }

  function stopDrawing(e) {
    if (!isDrawing) return;
    isDrawing = false;
    ctx.closePath();
    saveCanvasState();
  }

  // Bind Mouse & Touch Event Listeners
  canvas.addEventListener('mousedown', startDrawing, { passive: false });
  canvas.addEventListener('mousemove', draw, { passive: false });
  canvas.addEventListener('mouseup', stopDrawing, { passive: false });
  canvas.addEventListener('mouseleave', stopDrawing, { passive: false });

  canvas.addEventListener('touchstart', startDrawing, { passive: false });
  canvas.addEventListener('touchmove', draw, { passive: false });
  canvas.addEventListener('touchend', stopDrawing, { passive: false });
  canvas.addEventListener('touchcancel', stopDrawing, { passive: false });

  // Pencil Tool Button Handler
  if (penBtn) {
    penBtn.addEventListener('click', (e) => {
      e.preventDefault();
      hidePrompt();
      setDrawingStyles();
    });
  }

  // Top Right Reset Button
  if (undoBtn) {
    undoBtn.addEventListener('click', (e) => {
      e.preventDefault();
      clearEverything();
    });
  }

  function clearEverything() {
    isDrawing = false;
    hasPredicted = false;
    const rect = canvas.getBoundingClientRect();
    
    ctx.fillStyle = '#FFFFFF';
    ctx.fillRect(0, 0, rect.width, rect.height);
    
    setDrawingStyles();

    undoStack = [];
    saveCanvasState();

    resetToInitialState();
    showPrompt();
  }

  function resetToInitialState() {
    digitTicker.textContent = '-';
    topScoreText.textContent = '0.000000';
    
    const initialProbs = {};
    for (let i = 0; i <= 9; i++) {
      initialProbs[i.toString()] = 0.0;
    }
    renderRawProbabilityList(initialProbs, -1);
  }

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') predictDigit();
  });

  if (predictBtn) {
    predictBtn.addEventListener('click', (e) => {
      e.preventDefault();
      predictDigit();
    });
  }

  // 3. Prediction Engine (Live Render API)
  let isPredicting = false;

  async function predictDigit() {
    if (isPredicting) return;

    isPredicting = true;
    predictBtn.disabled = true;
    predictBtn.classList.add('opacity-50');

    const dataUrl = canvas.toDataURL('image/png');

    digitTicker.classList.remove('animate-pop-digit');

    // Live processing ticker animation while waiting for backend model response
    const tickerInterval = setInterval(() => {
      digitTicker.textContent = Math.floor(Math.random() * 10);
      topScoreText.textContent = (Math.random()).toFixed(6);

      const randomProbs = {};
      for (let i = 0; i <= 9; i++) {
        randomProbs[i.toString()] = Math.random() * 0.15;
      }
      renderRawProbabilityList(randomProbs, -1, true);
    }, 45);

    try {
      const response = await fetch(`${BACKEND_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: dataUrl })
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server Error ${response.status}`);
      }

      const data = await response.json();

      await new Promise(res => setTimeout(res, 350));
      clearInterval(tickerInterval);

      // Render actual model prediction
      digitTicker.textContent = data.digit;
      digitTicker.classList.add('animate-pop-digit');
      topScoreText.textContent = data.confidence.toFixed(6);
      backendStatusBadge.textContent = 'Render API Active';

      renderRawProbabilityList(data.probabilities, data.digit, false);
      hasPredicted = true;

    } catch (err) {
      clearInterval(tickerInterval);
      console.error('API Error:', err);

      digitTicker.textContent = 'ERR';
      topScoreText.textContent = '0.000000';
      backendStatusBadge.textContent = 'Render API Error';
      
      resetToInitialState();
      alert(`API Error: ${err.message || 'Unable to connect to model server.'}`);

    } finally {
      isPredicting = false;
      predictBtn.disabled = false;
      predictBtn.classList.remove('opacity-50');
    }
  }

  // 4. Render Raw Mathematical Probabilities for Digits 0 through 9
  function renderRawProbabilityList(probabilities, topDigit, isProcessing = false) {
    probabilityList.innerHTML = '';

    for (let i = 0; i <= 9; i++) {
      const key = i.toString();
      const rawVal = probabilities[key] !== undefined ? probabilities[key] : 0.0;
      const formattedFloat = rawVal.toFixed(6);
      const isTop = i === topDigit;
      
      const barPercent = Math.min(100, Math.max(0, rawVal * 100));

      const row = document.createElement('div');
      row.className = `flex items-center justify-between gap-3 py-1 px-2 rounded transition-all ${
        isTop ? 'bg-black text-white font-bold' : 'text-neutral-800 hover:bg-neutral-100'
      }`;

      row.innerHTML = `
        <!-- Digit Label -->
        <span class="w-6 text-left font-mono font-bold text-xs">${i}</span>
        
        <!-- Progress Bar -->
        <div class="flex-1 bg-neutral-200 rounded-full h-2 overflow-hidden ${isTop ? 'bg-neutral-800' : ''}">
          <div class="h-full rounded-full transition-all duration-200 ${isTop ? 'bg-white' : 'bg-black'}" style="width: ${isProcessing ? barPercent + '%' : '0%'}"></div>
        </div>
        
        <!-- Raw Float Score Count -->
        <span class="w-24 text-right font-mono text-[11px] whitespace-nowrap ${isTop ? 'text-white font-bold' : 'text-neutral-600'}">
          ${formattedFloat}
        </span>
      `;

      probabilityList.appendChild(row);

      if (!isProcessing) {
        setTimeout(() => {
          const bar = row.querySelector('.h-full');
          if (bar) bar.style.width = `${barPercent}%`;
        }, 20 + i * 12);
      }
    }
  }

  // 5. Initial App Setup
  initCanvas();
  resetToInitialState();

  // Check Live Render Backend Health
  fetch(`${BACKEND_URL}/health`)
    .then(res => res.json())
    .then(data => {
      if (data.model_loaded) {
        backendStatusBadge.textContent = 'Render API Live';
      } else {
        backendStatusBadge.textContent = 'Render API Active (No Model File)';
      }
    })
    .catch(() => {
      backendStatusBadge.textContent = 'Render API Waking Up...';
    });

  // Fade out loader
  setTimeout(() => {
    loaderOverlay.classList.add('opacity-0', 'pointer-events-none');
    setTimeout(() => {
      loaderOverlay.classList.add('hidden');
    }, 500);
  }, 400);
});
