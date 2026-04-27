import { SignJWT } from "jose";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "https://nevup-api-zq85.onrender.com";
const JWT_SECRET = process.env.NEXT_PUBLIC_JWT_SECRET ?? "";

const SECRET_KEY = new TextEncoder().encode(JWT_SECRET);

export const api = { url: API_URL };

/**
 * Mint a 1-hour HS256 JWT for a given userId. The brief's HS256 secret is shared knowledge
 * for the demo; in production, this would be a backend /login route.
 */
export async function mintToken(userId: string, name?: string): Promise<string> {
  return await new SignJWT({ role: "trader", name: name ?? "demo" })
    .setProtectedHeader({ alg: "HS256", typ: "JWT" })
    .setSubject(userId)
    .setIssuedAt()
    .setExpirationTime("1h")
    .sign(SECRET_KEY);
}

export type FetchOpts = RequestInit & { token?: string };

export async function apiFetch(path: string, opts: FetchOpts = {}): Promise<Response> {
  const headers = new Headers(opts.headers);
  if (opts.token) headers.set("Authorization", `Bearer ${opts.token}`);
  if (opts.body && !headers.has("Content-Type")) headers.set("Content-Type", "application/json");
  return fetch(`${API_URL}${path}`, { ...opts, headers });
}

export async function apiJson<T = unknown>(path: string, opts: FetchOpts = {}): Promise<T> {
  const r = await apiFetch(path, opts);
  if (!r.ok) {
    const text = await r.text();
    throw new ApiError(r.status, text);
  }
  return (await r.json()) as T;
}

export class ApiError extends Error {
  constructor(
    public status: number,
    public body: string,
  ) {
    super(`API ${status}`);
  }
}

/** Stream SSE from a POST request. Calls onToken for each `data:` frame, onDone when terminated. */
export async function streamSSE(
  path: string,
  body: unknown,
  token: string,
  handlers: { onToken: (t: string) => void; onDone: () => void; onError: (e: Error) => void },
  signal?: AbortSignal,
): Promise<void> {
  try {
    const r = await fetch(`${API_URL}${path}`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
        Accept: "text/event-stream",
      },
      body: JSON.stringify(body),
      signal,
    });
    if (!r.ok || !r.body) {
      const errBody = await r.text().catch(() => "");
      throw new ApiError(r.status, errBody);
    }
    const reader = r.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      // SSE frames are separated by blank lines (\n\n).
      const frames = buffer.split("\n\n");
      buffer = frames.pop() ?? "";
      for (const frame of frames) {
        if (!frame.trim()) continue;
        const dataLine = frame
          .split("\n")
          .find((l) => l.startsWith("data:"));
        if (!dataLine) continue;
        // SSE spec: strip EXACTLY one optional space after the colon.
        // .trimStart() would strip meaningful leading spaces from token deltas
        // like " see", " you've", causing concatenated output ("Iseeyou've...").
        let payload = dataLine.slice(5);
        if (payload.startsWith(" ")) payload = payload.slice(1);
        if (payload === "[DONE]") {
          handlers.onDone();
          return;
        }
        handlers.onToken(payload);
      }
    }
    handlers.onDone();
  } catch (e) {
    if ((e as Error).name === "AbortError") return;
    handlers.onError(e as Error);
  }
}
