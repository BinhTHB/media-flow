"""One-time OAuth setup for the YouTube publish target.

Opens the Google consent URL, catches the redirect on a local loopback
server, exchanges the authorization code and prints the env lines to add
to .env. Run: python scripts/get_youtube_refresh_token.py
Requires YOUTUBE_CLIENT_ID / YOUTUBE_CLIENT_SECRET (env or .env).
"""
import json
import os
import sys
import threading
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REDIRECT_PORT = 8765
REDIRECT_URI = f"http://localhost:{REDIRECT_PORT}"
SCOPE = "https://www.googleapis.com/auth/youtube.upload"


def load_env(path: str) -> dict:
    env = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def main() -> int:
    file_env = load_env(os.path.join(ROOT, ".env"))
    client_id = os.environ.get("YOUTUBE_CLIENT_ID") or file_env.get("YOUTUBE_CLIENT_ID")
    client_secret = os.environ.get("YOUTUBE_CLIENT_SECRET") or file_env.get("YOUTUBE_CLIENT_SECRET")
    if not client_id or not client_secret:
        print("Missing YOUTUBE_CLIENT_ID / YOUTUBE_CLIENT_SECRET.")
        print("Google Cloud Console -> APIs & Services -> Credentials -> OAuth client (Desktop app).")
        print("Then: set YOUTUBE_CLIENT_ID=... and YOUTUBE_CLIENT_SECRET=..., or add them to .env")
        return 1

    params = urllib.parse.urlencode({
        "client_id": client_id,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": SCOPE,
        "access_type": "offline",
        "prompt": "consent",
    })
    consent_url = f"https://accounts.google.com/o/oauth2/v2/auth?{params}"

    captured = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            captured["code"] = (qs.get("code") or [None])[0]
            captured["error"] = (qs.get("error") or [None])[0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write("<h1>OK - return to the terminal.</h1>".encode("utf-8"))

        def log_message(self, *args):
            pass

    server = HTTPServer(("localhost", REDIRECT_PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    print("1. Log in with the Google account that owns the target channel.")
    print("2. Approve the consent screen.\n")
    print(consent_url, "\n")
    try:
        import webbrowser
        webbrowser.open(consent_url)
    except Exception:
        pass

    for _ in range(300):
        if captured.get("code") or captured.get("error"):
            break
        threading.Event().wait(1.0)
    server.shutdown()

    if captured.get("error"):
        print(f"Consent failed: {captured['error']}")
        return 1
    if not captured.get("code"):
        print("Timed out waiting for the consent redirect.")
        return 1

    data = urllib.parse.urlencode({
        "code": captured["code"],
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": REDIRECT_URI,
        "grant_type": "authorization_code",
    }).encode()
    req = urllib.request.Request("https://oauth2.googleapis.com/token", data=data)
    with urllib.request.urlopen(req, timeout=30) as resp:
        tokens = json.loads(resp.read().decode())

    refresh = tokens.get("refresh_token")
    if not refresh:
        print("No refresh_token in response (account already consented before?).")
        print(json.dumps({k: v for k, v in tokens.items() if k != "access_token"}, indent=2))
        print("Re-run with prompt=consent (already set) or remove app access at "
              "https://myaccount.google.com/permissions and retry.")
        return 1

    print("\nAdd these lines to .env:\n")
    print(f"YOUTUBE_CLIENT_ID={client_id}")
    print(f"YOUTUBE_CLIENT_SECRET={client_secret}")
    print(f"YOUTUBE_REFRESH_TOKEN={refresh}")

    data = urllib.parse.urlencode({
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh,
        "grant_type": "refresh_token",
    }).encode()
    req = urllib.request.Request("https://oauth2.googleapis.com/token", data=data)
    with urllib.request.urlopen(req, timeout=30) as resp:
        check = json.loads(resp.read().decode())
    print("\nToken check:", "OK (access token issued)" if check.get("access_token") else check)
    return 0 if check.get("access_token") else 1


if __name__ == "__main__":
    sys.exit(main())
