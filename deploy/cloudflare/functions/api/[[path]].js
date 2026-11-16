// Same-origin proxy for the page's /api/* calls (issue #25).
//
//   browser --(Access login, Pages app)--> this Function --(proxy credential)--> public entry of the VM
//           --> app/backend.py on the GCE VM (127.0.0.1)
//
// The page keeps calling relative /api/... paths, so there is no CORS and no third-party cookie. The backend
// checks the proxy credential (app/access.py). Public mode (default): anyone may use the page. With the optional
// Access sign-in, the user's token for this Pages site is forwarded in X-Agentforge-User-Jwt and checked too.
//
// Settings (Pages project -> Settings -> Variables and secrets):
//   API_ORIGIN               free setup: https://<vm>.<tailnet>.ts.net (Tailscale Funnel)
//                            own domain: https://api.<your domain> (Cloudflare Tunnel)
//   PROXY_SECRET             free setup: same value as AGENTFORGE_PROXY_SECRET on the VM (secret)
//   CF_ACCESS_CLIENT_ID      own domain: service token ID (secret)
//   CF_ACCESS_CLIENT_SECRET  own domain: service token secret (secret)

const FORWARDED = ["content-type", "accept"];

function userToken(request) {
  const header = request.headers.get("Cf-Access-Jwt-Assertion");
  if (header) return header;
  const cookie = request.headers.get("Cookie") || "";
  const match = cookie.match(/(?:^|;\s*)CF_Authorization=([^;]+)/);
  return match ? match[1] : null;
}

function jsonError(status, code, message) {
  return new Response(JSON.stringify({ error: { code, message } }), {
    status, headers: { "Content-Type": "application/json; charset=utf-8", "Cache-Control": "no-store" },
  });
}

export async function onRequest({ request, env }) {
  const serviceToken = env.CF_ACCESS_CLIENT_ID && env.CF_ACCESS_CLIENT_SECRET;
  if (!env.API_ORIGIN || (!env.PROXY_SECRET && !serviceToken)) {
    return jsonError(500, "not_configured", "The API proxy is not configured.");
  }
  if (!["GET", "POST"].includes(request.method)) {
    return jsonError(405, "method_not_allowed", "Use GET or POST.");
  }
  // Public mode has no sign-in, so there may be no user token; with Access on, the backend requires it.
  const token = userToken(request);

  const url = new URL(request.url);
  const target = new URL(url.pathname + url.search, env.API_ORIGIN);
  const headers = new Headers();
  for (const name of FORWARDED) {
    const value = request.headers.get(name);
    if (value) headers.set(name, value);
  }
  if (env.PROXY_SECRET) {
    headers.set("X-Agentforge-Proxy-Secret", env.PROXY_SECRET);
  } else {
    headers.set("CF-Access-Client-Id", env.CF_ACCESS_CLIENT_ID);
    headers.set("CF-Access-Client-Secret", env.CF_ACCESS_CLIENT_SECRET);
  }
  if (token) headers.set("X-Agentforge-User-Jwt", token);
  // The visitor's address as Cloudflare saw it: the backend's per-visitor limits use it (trusted only together
  // with the proxy secret, so a browser cannot set it).
  const ip = request.headers.get("CF-Connecting-IP");
  if (ip) headers.set("X-Agentforge-Client-Ip", ip);

  let upstream;
  try {
    upstream = await fetch(target, {
      method: request.method, headers, redirect: "manual",
      body: request.method === "POST" ? request.body : undefined,
    });
  } catch (e) {
    return jsonError(503, "harness_unavailable", "The shopping service is not reachable right now.");
  }
  if (upstream.status >= 300 && upstream.status < 400) {
    // Access redirected instead of accepting the service token: a configuration problem, not the user's.
    return jsonError(502, "proxy_rejected", "The API rejected the proxy's credentials.");
  }
  // Stream the body through unchanged (SSE for /api/ask and /api/answer). A browser that leaves aborts this
  // request, the backend sees the broken stream and cancels the run.
  const out = new Headers(upstream.headers);
  out.set("Cache-Control", "no-store");
  return new Response(upstream.body, { status: upstream.status, headers: out });
}
