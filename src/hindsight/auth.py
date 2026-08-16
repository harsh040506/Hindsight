"""
OAuth credential loading and API client construction.

Hindsight is deliberately permissive about *where* credentials come from,
because most people running automated channels already have working OAuth
tokens minted by their upload script. Re-authorising every channel just to
read stats is a pointless chore, so Hindsight reuses those tokens if they
carry a read scope, and only offers a login flow when there is nothing to
reuse.

It is deliberately strict about what it does with them: read endpoints only.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from . import config

log = logging.getLogger(__name__)

# Any one of these grants enough access to read video statistics.
_READ_SCOPES = {
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/youtube",
    "https://www.googleapis.com/auth/youtube.force-ssl",
}


class AuthError(RuntimeError):
    """Raised when credentials are missing, unusable, or lack read access."""


@dataclass
class Session:
    """
    A ready-to-use YouTube client plus the channel it is pointed at.

    Two modes produce one of these:

      OAuth mode      `credentials` is set and `channel_ref` is None. The
                      session reads the authenticated user's own channel, and
                      can see unlisted/private videos.

      API-key mode    `credentials` is None and `channel_ref` names a public
                      channel. No OAuth, no consent screen, no token file --
                      just a key from the Cloud console. Sees only what the
                      public sees, which for this analysis is everything that
                      matters, since unpublished videos are excluded anyway.

    API-key mode exists because the analysis works just as well on someone
    else's channel as on your own. Being able to point Hindsight at any public
    channel turns it from a tool you set up into a tool you try.
    """

    youtube: Any
    slug: str
    credentials: Credentials | None = None
    token_path: Path | None = None
    channel_ref: str | None = None
    analytics: Any | None = None
    analytics_error: str | None = None

    @property
    def has_analytics(self) -> bool:
        """True if the richer retention/CTR endpoints are actually usable."""
        return self.analytics is not None

    @property
    def is_public_mode(self) -> bool:
        return self.credentials is None


def load_credentials(token_path: Path) -> Credentials:
    """
    Load and, if needed, refresh OAuth credentials from a token file.

    Refreshed tokens are written back so the next run starts valid. If the file
    is not writable that is not fatal -- we log and continue with the in-memory
    credentials.
    """
    if not token_path.is_file():
        raise AuthError(f"No token file at {token_path}")

    try:
        creds = Credentials.from_authorized_user_file(str(token_path))
    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        raise AuthError(f"{token_path} is not a valid OAuth token file: {exc}") from exc

    granted = set(creds.scopes or [])
    if not (granted & _READ_SCOPES):
        raise AuthError(
            f"Token {token_path.name} carries scopes {sorted(granted)}, none of "
            f"which allow reading channel data. Hindsight needs "
            f"'youtube.readonly'. Run `hindsight auth login` to mint one."
        )

    if not creds.valid:
        if not creds.refresh_token:
            raise AuthError(
                f"Token {token_path.name} is expired and has no refresh token. "
                f"Run `hindsight auth login` to re-authorise."
            )
        log.debug("Refreshing expired credentials for %s", token_path.name)
        creds.refresh(Request())
        try:
            token_path.write_text(creds.to_json(), encoding="utf-8")
        except OSError as exc:  # pragma: no cover - filesystem dependent
            log.warning("Could not persist refreshed token to %s: %s", token_path, exc)

    return creds


def open_session(
    token_path: Path, slug: str | None = None, probe_analytics: bool = True
) -> Session:
    """
    Build API clients from a token file.

    The Analytics client is constructed optimistically and then *probed* with a
    cheap real query, because holding the scope is not sufficient -- the
    YouTube Analytics API also has to be enabled on the Cloud project behind
    the OAuth client. Probing here means the rest of the codebase can trust
    `session.has_analytics` instead of discovering the problem mid-report.
    """
    creds = load_credentials(token_path)
    youtube = build(
        config.YOUTUBE_API_SERVICE_NAME,
        config.YOUTUBE_API_VERSION,
        credentials=creds,
        cache_discovery=False,
    )

    session = Session(
        credentials=creds,
        youtube=youtube,
        slug=slug or token_path.stem.replace("_token", ""),
        token_path=token_path,
    )

    if probe_analytics:
        _attach_analytics(session)

    return session


def open_public_session(api_key: str, channel_ref: str) -> Session:
    """
    Build a read-only client for any public channel using only an API key.

    `channel_ref` accepts whatever the user is likely to have to hand:
    a channel id (`UC...`), a handle (`@mkbhd`), or a channel URL. It is
    normalised here rather than at the call site so every entry point behaves
    identically.
    """
    if not api_key:
        raise AuthError(
            "No API key. Set YOUTUBE_API_KEY in your environment or pass "
            "--api-key. Create one at console.cloud.google.com -> APIs & "
            "Services -> Credentials -> Create credentials -> API key, then "
            "enable 'YouTube Data API v3'. No OAuth consent screen is needed."
        )

    youtube = build(
        config.YOUTUBE_API_SERVICE_NAME,
        config.YOUTUBE_API_VERSION,
        developerKey=api_key,
        cache_discovery=False,
    )
    ref = normalize_channel_ref(channel_ref)
    return Session(youtube=youtube, slug=_slug_from_ref(ref), channel_ref=ref)


def normalize_channel_ref(ref: str) -> str:
    """
    Reduce a user-supplied channel reference to an id or a handle.

    Accepts:
        UCxxxxxxxxxxxxxxxxxxxxxx
        @some_handle
        https://www.youtube.com/@some_handle
        https://www.youtube.com/channel/UCxxxxxxxxxxxxxxxxxxxxxx
        youtube.com/c/SomeName          (treated as a handle)
    """
    ref = (ref or "").strip()
    if not ref:
        raise AuthError("Empty channel reference.")

    if "youtube.com" in ref or "youtu.be" in ref:
        path = ref.split("youtube.com", 1)[-1].split("?", 1)[0].strip("/")
        parts = [p for p in path.split("/") if p]
        if not parts:
            raise AuthError(f"Could not read a channel from URL: {ref}")
        if parts[0] == "channel" and len(parts) > 1:
            return parts[1]
        if parts[0] in ("c", "user") and len(parts) > 1:
            return "@" + parts[1].lstrip("@")
        return "@" + parts[0].lstrip("@")

    return ref


def _slug_from_ref(ref: str) -> str:
    return ref.lstrip("@") if ref.startswith("@") else ref


def _attach_analytics(session: Session) -> None:
    """Attach an Analytics client if the scope is held and the API is enabled."""
    granted = set(session.credentials.scopes or [])
    if config.ANALYTICS_SCOPE not in granted:
        session.analytics_error = (
            "token does not carry the yt-analytics.readonly scope"
        )
        return

    try:
        client = build(
            config.ANALYTICS_API_SERVICE_NAME,
            config.ANALYTICS_API_VERSION,
            credentials=session.credentials,
            cache_discovery=False,
        )
        # Cheap one-day probe. If the API is disabled on the project this
        # raises 403 with reason 'accessNotConfigured'.
        client.reports().query(
            ids="channel==MINE",
            startDate="2020-01-01",
            endDate="2020-01-02",
            metrics="views",
        ).execute()
    except HttpError as exc:
        session.analytics_error = f"Analytics API unavailable: {_http_reason(exc)}"
        return
    except Exception as exc:  # pragma: no cover - defensive
        session.analytics_error = f"Analytics API unavailable: {exc}"
        return

    session.analytics = client


def _http_reason(exc: HttpError) -> str:
    """Extract the human-meaningful reason from a Google API error."""
    try:
        payload = json.loads(exc.content.decode("utf-8"))
        err = payload.get("error", {})
        details = err.get("errors") or []
        reason = details[0].get("reason") if details else None
        message = err.get("message", "")
        return f"{reason or exc.resp.status}: {message[:200]}"
    except Exception:  # pragma: no cover - malformed error body
        return str(exc)[:200]


def login(client_secrets: Path | None = None, token_out: Path | None = None) -> Path:
    """
    Run the interactive OAuth flow and write a new token file.

    Only needed when you have no existing token to reuse. Opens a browser and
    starts a temporary local server to catch the redirect, so this requires an
    interactive desktop session.
    """
    from google_auth_oauthlib.flow import InstalledAppFlow

    secrets = client_secrets or config.CLIENT_SECRETS_FILE
    if not secrets.is_file():
        raise AuthError(
            f"OAuth client config not found at {secrets}. Download it from the "
            f"Google Cloud console (APIs & Services -> Credentials -> OAuth "
            f"client ID -> Desktop app) and save it there, or pass "
            f"--client-secrets. See docs/SETUP.md."
        )

    scopes = config.YOUTUBE_SCOPES + [config.ANALYTICS_SCOPE]
    flow = InstalledAppFlow.from_client_secrets_file(str(secrets), scopes)
    creds = flow.run_local_server(port=0)

    out = token_out or Path("hindsight_token.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(creds.to_json(), encoding="utf-8")
    return out
