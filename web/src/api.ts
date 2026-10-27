let token = '';
export function setToken(value: string) { token = value; }
export async function api<T = any>(path: string, body?: unknown, method?: string): Promise<T> {
  const response = await fetch('/api' + path, {
    method: method || (body === undefined ? 'GET' : 'POST'),
    headers: { ...(body instanceof FormData ? {} : body !== undefined ? {'Content-Type': 'application/json'} : {}), 'X-Draftfit-Token': token },
    body: body === undefined ? undefined : body instanceof FormData ? body : JSON.stringify(body),
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({detail: 'Cannot connect to the server.'}));
    throw new Error(typeof error.detail === 'string' ? error.detail : error.detail?.map((e: any) => `${e.loc?.slice(1).join('.')}: ${e.msg}`).join('\n') || 'The request could not be completed.');
  }
  return response.json();
}
export function downloadJSON(value: unknown, name: string) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], {type:'application/json'}));
  const a = document.createElement('a'); a.href = url; a.download = name; a.click(); URL.revokeObjectURL(url);
}
