// LangShark panel — global helpers
(function () {
  // ────── Toast (acessível: role=status + aria-live no host) ──────
  function showToast(message, kind) {
    const host = document.getElementById('toast-host');
    if (!host) return;
    const t = document.createElement('div');
    t.className = 'toast' + (kind ? ' toast-' + kind : '');
    t.textContent = message;
    if (kind === 'error') t.setAttribute('role', 'alert');
    host.appendChild(t);
    setTimeout(() => {
      t.style.opacity = '0';
      t.style.transition = 'opacity 200ms';
      setTimeout(() => t.remove(), 220);
    }, 3500);
  }
  window.showToast = showToast;

  // ────── Focus trap pra modais (Tab cycle dentro do modal) ──────
  function trapFocus(container) {
    if (!container) return () => {};
    const selector = 'a[href], button:not([disabled]), textarea:not([disabled]), input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])';
    function handler(e) {
      if (e.key !== 'Tab') return;
      const focusables = Array.from(container.querySelectorAll(selector))
        .filter(el => !el.hasAttribute('disabled') && el.offsetParent !== null);
      if (focusables.length === 0) return;
      const first = focusables[0];
      const last = focusables[focusables.length - 1];
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault(); last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault(); first.focus();
      }
    }
    container.addEventListener('keydown', handler);
    return () => container.removeEventListener('keydown', handler);
  }
  window.trapFocus = trapFocus;

  // ────── Auto-aplica focus trap em modais Alpine quando aparecem ──────
  // Observa DOM por elementos com .modal-card visíveis
  const observer = new MutationObserver(() => {
    document.querySelectorAll('.modal-card').forEach(card => {
      if (card.dataset.trapAttached) return;
      const overlay = card.closest('.modal-overlay');
      if (overlay && overlay.style.display !== 'none' && overlay.offsetParent !== null) {
        card.dataset.trapAttached = '1';
        const release = trapFocus(card);
        // Libera quando modal desaparece
        const releaseObserver = new MutationObserver(() => {
          if (overlay.style.display === 'none' || overlay.offsetParent === null) {
            release();
            delete card.dataset.trapAttached;
            releaseObserver.disconnect();
          }
        });
        releaseObserver.observe(overlay, { attributes: true, attributeFilter: ['style'] });
      }
    });
  });
  observer.observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ['style'] });

  // ────── CSRF token (double-submit cookie) ──────
  // Lê cookie csrftoken e injeta header X-CSRF-Token em todo fetch mutating.
  function getCookie(name) {
    const m = document.cookie.match(new RegExp('(?:^|; )' + name + '=([^;]*)'));
    return m ? decodeURIComponent(m[1]) : '';
  }
  const _origFetch = window.fetch.bind(window);
  window.fetch = function (input, init) {
    init = init || {};
    const method = (init.method || 'GET').toUpperCase();
    if (['POST', 'PATCH', 'PUT', 'DELETE'].includes(method)) {
      const csrf = getCookie('csrftoken');
      if (csrf) {
        init.headers = Object.assign({}, init.headers || {}, { 'X-CSRF-Token': csrf });
      }
    }
    return _origFetch(input, init);
  };

  // ────── HTMX hooks ──────
  document.body.addEventListener('htmx:responseError', function (e) {
    showToast('Falha: ' + (e.detail.xhr.statusText || 'erro'), 'error');
  });
  document.body.addEventListener('htmx:afterRequest', function (e) {
    const xhr = e.detail.xhr;
    if (xhr.status >= 200 && xhr.status < 300 && e.detail.requestConfig.verb !== 'get') {
      const t = xhr.getResponseHeader('X-Toast');
      if (t) showToast(t, 'success');
    }
  });
  document.body.addEventListener('htmx:configRequest', function (e) {
    const csrf = getCookie('csrftoken');
    if (csrf) e.detail.headers['X-CSRF-Token'] = csrf;
  });
})();
