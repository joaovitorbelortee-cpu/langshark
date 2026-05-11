// LangShark panel — global helpers
(function () {
  function showToast(message, kind) {
    const host = document.getElementById('toast-host');
    if (!host) return;
    const t = document.createElement('div');
    t.className = 'toast' + (kind ? ' toast-' + kind : '');
    t.textContent = message;
    host.appendChild(t);
    setTimeout(() => {
      t.style.opacity = '0';
      t.style.transition = 'opacity 200ms';
      setTimeout(() => t.remove(), 220);
    }, 3500);
  }
  window.showToast = showToast;

  document.body.addEventListener('htmx:responseError', function (e) {
    showToast('Falha: ' + (e.detail.xhr.statusText || 'erro'), 'error');
  });
  document.body.addEventListener('htmx:afterRequest', function (e) {
    const xhr = e.detail.xhr;
    if (xhr.status >= 200 && xhr.status < 300 && e.detail.requestConfig.verb !== 'get') {
      // Toasts apenas em mutações ok
      const t = xhr.getResponseHeader('X-Toast');
      if (t) showToast(t, 'success');
    }
  });
})();
