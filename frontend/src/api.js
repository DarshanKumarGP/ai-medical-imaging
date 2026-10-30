const BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

export const imageUrl = (path) => (path ? `${BASE}${path}` : null);

async function request(path, options) {
  const res = await fetch(`${BASE}${path}`, options);
  if (!res.ok) {
    let message = res.statusText;
    try {
      const body = await res.json();
      message = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* keep statusText */
    }
    throw new Error(message);
  }
  return res.json();
}

export const getStats = () => request("/stats");
export const getStudies = () => request("/studies?limit=500");
export const getStudy = (id) => request(`/studies/${id}`);
export const postDecision = (id, body) =>
  request(`/studies/${id}/decision`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });