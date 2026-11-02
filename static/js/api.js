// Shared API access: optional bearer token and error-envelope decoding.

export const AUTH_TOKEN = (() => {
  const params = new URLSearchParams(window.location.search);
  const fromUrl = (params.get('token') || '').trim();
  if (fromUrl) {
    try { localStorage.setItem('vectorview_token', fromUrl); } catch (err) { /* storage unavailable */ }
    return fromUrl;
  }
  try { return (localStorage.getItem('vectorview_token') || '').trim(); } catch (err) { return ''; }
})();

export async function readApiError(res, fallback) {
  let text = '';
  try { text = await res.text(); } catch (err) { return fallback; }
  if (text) {
    try {
      const parsed = JSON.parse(text);
      const message = parsed && parsed.error && parsed.error.message;
      if (message) return message;
    } catch (err) { /* not a json envelope */ }
  }
  return text || fallback;
}
