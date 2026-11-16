"""Who may call the public backend (issue #25).

Two modes:
- ProxyAuth (default, public page, anyone may use it): only the Pages Function's shared secret is checked.
- AccessAuth (optional sign-in with Cloudflare Access): described below.

Request path:  browser --(Access login)--> Cloudflare Pages + Pages Function --(proxy credential)-->
               public entry to the VM --> this backend on 127.0.0.1

The signed-in user's Access token for the Pages application is forwarded by the Function in
`X-Agentforge-User-Jwt`; its `email` is the identity that owns runs. It is an RS256 JWT checked against
https://<team>.cloudflareaccess.com/cdn-cgi/access/certs (keys cached).

The request must also prove it came from the Function, in one of two ways:

- free setup (pages.dev + Tailscale Funnel, deploy/gcp/README.md): a shared secret in
  `X-Agentforge-Proxy-Secret`, known only to the Function (a Pages secret) and this host;
- own domain (Cloudflare Tunnel + an Access application on the API hostname whose only policy is the
  Function's service token): `Cf-Access-Jwt-Assertion`, added by Access, checked like the user token against
  the API application's audience. (Access replaces that header on this hop, which is why the user token
  travels in its own header.)

Needs PyJWT with cryptography (deploy/gcp/requirements-cloud.txt); the local backend does not import this.
"""
from __future__ import annotations

import hmac
import re

USER_HEADER = "X-Agentforge-User-Jwt"
EDGE_HEADER = "Cf-Access-Jwt-Assertion"
SECRET_HEADER = "X-Agentforge-Proxy-Secret"


class AuthError(Exception):
    pass


CLIENT_IP_HEADER = "X-Agentforge-Client-Ip"


def _check_secret(headers, secret: str) -> None:
    given = headers.get(SECRET_HEADER) or ""
    if not hmac.compare_digest(given.encode(), secret.encode()):
        raise AuthError("missing or wrong proxy secret")


def _valid_secret(secret: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]{32,}", secret or ""):
        raise ValueError("proxy_secret must be at least 32 letters, digits, '-' or '_' "
                         "(e.g. openssl rand -hex 32), not the example placeholder")
    return secret


class ProxyAuth:
    """Public mode: anyone may use the page, no sign-in. The backend only checks that the request came through
    the Pages Function (shared secret); runs belong to the page's session ID, and the visitor's IP that the
    Function forwards (Cloudflare's CF-Connecting-IP) is the key for the per-visitor limits."""

    def __init__(self, proxy_secret: str):
        self.proxy_secret = _valid_secret(proxy_secret)

    def identify(self, headers) -> None:
        _check_secret(headers, self.proxy_secret)
        return None

    def limit_key(self, headers) -> str | None:
        ip = (headers.get(CLIENT_IP_HEADER) or "").strip()
        return f"ip:{ip}" if re.fullmatch(r"[0-9A-Fa-f:.]{2,45}", ip) else None


class AccessAuth:
    def __init__(self, team_domain: str, user_aud: str, *, api_aud: str | None = None,
                 proxy_secret: str | None = None, key_for_token=None):
        if bool(api_aud) == bool(proxy_secret):
            raise ValueError("give exactly one proxy check: api_aud (own domain) or proxy_secret (free setup)")
        if proxy_secret is not None:
            _valid_secret(proxy_secret)
        import jwt
        self._jwt = jwt
        self.issuer = f"https://{team_domain.removeprefix('https://').rstrip('/')}"
        self.user_aud, self.api_aud, self.proxy_secret = user_aud, api_aud, proxy_secret
        if key_for_token is None:
            client = jwt.PyJWKClient(f"{self.issuer}/cdn-cgi/access/certs", cache_keys=True, lifespan=3600)
            key_for_token = lambda token: client.get_signing_key_from_jwt(token).key  # noqa: E731
        self.key_for_token = key_for_token

    def _claims(self, token: str | None, audience: str, what: str) -> dict:
        if not token:
            raise AuthError(f"missing {what} token")
        try:
            return self._jwt.decode(token, self.key_for_token(token), algorithms=["RS256"], audience=audience,
                                    issuer=self.issuer, options={"require": ["exp", "iat", "aud", "iss"]})
        except Exception as exc:                          # expired, wrong audience/issuer, bad signature, no key
            raise AuthError(f"invalid {what} token: {type(exc).__name__}") from None

    def _check_proxy(self, headers) -> None:
        if self.proxy_secret is not None:
            _check_secret(headers, self.proxy_secret)
        else:
            self._claims(headers.get(EDGE_HEADER), self.api_aud, "edge")

    def identify(self, headers) -> str:
        """The signed-in user's email; AuthError unless the proxy check and the user token both pass."""
        self._check_proxy(headers)
        user = self._claims(headers.get(USER_HEADER), self.user_aud, "user")
        email = user.get("email")
        if not isinstance(email, str) or "@" not in email:
            raise AuthError("user token has no email")
        return email.lower()
