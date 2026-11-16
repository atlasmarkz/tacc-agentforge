"""Cloudflare Access identity for the public backend (issue #25).

Request path:  browser --(Access login)--> Cloudflare Pages + Pages Function --(Access service token)-->
               Cloudflare Tunnel --> this backend on 127.0.0.1

Two Access applications sign two JWTs with the team's keys:

- `Cf-Access-Jwt-Assertion`: added by Access in front of the API hostname. Its audience is the API application,
  whose only policy is the Pages Function's service token, so a valid one proves the request came through
  Access and the Function (the backend also only listens on 127.0.0.1, behind cloudflared).
- `X-Agentforge-User-Jwt`: the signed-in user's token for the Pages application, forwarded by the Function
  (Access overwrites `Cf-Access-Jwt-Assertion` on the second hop, so the user token travels in its own
  header). Its `email` is the identity that owns runs.

Both are RS256 JWTs checked against https://<team>.cloudflareaccess.com/cdn-cgi/access/certs (cached).
Needs PyJWT with cryptography (deploy/gcp/requirements-cloud.txt); the local backend does not import this.
"""
from __future__ import annotations

USER_HEADER = "X-Agentforge-User-Jwt"
EDGE_HEADER = "Cf-Access-Jwt-Assertion"


class AuthError(Exception):
    pass


class AccessAuth:
    def __init__(self, team_domain: str, api_aud: str, user_aud: str, *, key_for_token=None):
        import jwt
        self._jwt = jwt
        self.issuer = f"https://{team_domain.removeprefix('https://').rstrip('/')}"
        self.api_aud, self.user_aud = api_aud, user_aud
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

    def identify(self, headers) -> str:
        """The signed-in user's email; AuthError unless both tokens are valid."""
        self._claims(headers.get(EDGE_HEADER), self.api_aud, "edge")
        user = self._claims(headers.get(USER_HEADER), self.user_aud, "user")
        email = user.get("email")
        if not isinstance(email, str) or "@" not in email:
            raise AuthError("user token has no email")
        return email.lower()
