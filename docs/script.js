(function () {
  // RAG backend URL: same-origin only when page is served over http(s) from the app (e.g. http://localhost:5000).
  // Otherwise use localhost:5000 so opening file:// still tries the right backend (browser may block it).
  var apiOrigin = (typeof window !== 'undefined' && window.location && (window.location.protocol === 'http:' || window.location.protocol === 'https:') && window.location.host) ? window.location.origin : 'http://localhost:5000';
  const RAG_API_URL = (typeof window !== 'undefined' && window.CAPC_RAG_API_URL) || (apiOrigin + '/api/chat');
  const RAG_HEALTH_URL = apiOrigin + '/api/health';

  function getBackendOfflineMessage() {
    if (typeof window !== 'undefined' && (window.location.protocol === 'file:' || !window.location.host)) {
      return 'You opened this page as a file. The chat only works when you open it from the running app. Run in terminal: python run_local.py  then in your browser go to: http://localhost:5000';
    }
    return 'Backend not reachable. From the project root run: python run_local.py  then open http://localhost:5000 in the browser.';
  }

  const messagesEl = document.getElementById('messages');
  const placeholderEl = document.getElementById('placeholder');
  const userInputEl = document.getElementById('userInput');
  const sendBtn = document.getElementById('sendBtn');
  const errorMsgEl = document.getElementById('errorMsg');

  addMessage('assistant', "Hello! I'm the CAPC AI Assistant, powered by CAPC guidelines (RAG). I can help with parasite guidelines, prevention, and veterinary resources. How can I assist you today?");

  // Check if backend is reachable on load; show banner if not (e.g. file:// or backend not running)
  (function checkBackend() {
    var banner = document.getElementById('backendBanner');
    if (!banner) return;
    fetch(RAG_HEALTH_URL, { method: 'GET', cache: 'no-store' })
      .then(function (r) { if (r.ok) banner.style.display = 'none'; })
      .catch(function () {
        banner.style.display = 'block';
        banner.innerHTML = 'Backend not running. Run: <code>python run_local.py</code> then open <a href="http://localhost:5000">http://localhost:5000</a> in this browser.';
      });
  })();

  function hidePlaceholder() {
    if (placeholderEl) placeholderEl.style.display = 'none';
  }

  function showError(msg) {
    errorMsgEl.textContent = msg;
    errorMsgEl.style.display = 'block';
    setTimeout(function () {
      errorMsgEl.style.display = 'none';
    }, 6000);
  }

  // Make URLs in text clickable and preserve line breaks
  function linkify(text) {
    const urlRe = /(https?:\/\/[^\s)]+)/g;
    const parts = text.split(urlRe);
    if (parts.length === 1) return text;
    return parts.map(function (p) {
      if (p.indexOf('http') === 0) return '<a href="' + p + '" target="_blank" rel="noopener">' + p + '</a>';
      return p;
    }).join('');
  }

  function formatAssistantMessage(text) {
    if (!text) return '';
    var escaped = text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    return linkify(escaped).replace(/\n/g, '<br>');
  }

  function addMessage(role, text) {
    hidePlaceholder();
    const div = document.createElement('div');
    div.className = 'message ' + role;
    div.innerHTML = '<div class="label">' + (role === 'user' ? 'You' : 'CAPC Assistant') + '</div><div class="text"></div>';
    var textEl = div.querySelector('.text');
    if (role === 'assistant') {
      textEl.innerHTML = formatAssistantMessage(text);
    } else {
      textEl.textContent = text;
    }
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return div;
  }

  function addTypingIndicator() {
    hidePlaceholder();
    const div = document.createElement('div');
    div.className = 'message assistant';
    div.id = 'typingIndicator';
    div.innerHTML = '<div class="label">CAPC Assistant</div><div class="typing"><span></span><span></span><span></span></div>';
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return div;
  }

  function removeTypingIndicator() {
    const el = document.getElementById('typingIndicator');
    if (el) el.remove();
  }

  function setLoading(loading) {
    sendBtn.disabled = loading;
    userInputEl.disabled = loading;
  }

  userInputEl.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendBtn.click();
    }
  });

  sendBtn.addEventListener('click', function () {
    const query = (userInputEl.value || '').trim();
    if (!query) return;

    addMessage('user', query);
    userInputEl.value = '';
    addTypingIndicator();
    setLoading(true);
    errorMsgEl.style.display = 'none';

    fetch(RAG_API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: query })
    })
      .then(function (res) {
        return res.json().then(function (data) {
          if (!res.ok) throw new Error(data.error || data.message || 'Request failed');
          return data;
        });
      })
      .then(function (data) {
        removeTypingIndicator();
        const text = data.response;
        if (text) {
          addMessage('assistant', text.trim());
        } else {
          addMessage('assistant', 'No response was generated. Please try again.');
        }
      })
      .catch(function (err) {
        removeTypingIndicator();
        var isNetwork = (err.name === 'TypeError' && err.message && err.message.indexOf('fetch') !== -1) || (err.message && (err.message.indexOf('Failed') !== -1 || err.message.indexOf('NetworkError') !== -1));
        if (isNetwork) {
          showError('Cannot reach the RAG backend.');
          addMessage('assistant', getBackendOfflineMessage());
        } else {
          showError(err.message || 'Something went wrong. Please try again.');
          addMessage('assistant', 'Backend error: ' + (err.message || 'Something went wrong. Check the terminal where you ran python run_local.py for details.'));
        }
      })
      .finally(function () {
        setLoading(false);
        userInputEl.focus();
      });
  });
})();
