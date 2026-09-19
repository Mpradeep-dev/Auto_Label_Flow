"""Shared, retry-hardened wrapper around Roboflow's `/search` endpoint.
Originally lived only in `roboflow_import.py` (pulling images in); moved
here once `roboflow_export.py` needed the identical retry/error-envelope
handling to discover which batch an already-known image currently belongs
to, before carving it into a review job (see
`roboflow_export._discover_batch_membership`)."""
from __future__ import annotations

import logging
import time

import requests

logger = logging.getLogger(__name__)

# Roboflow's /search occasionally throws a transient 5xx (observed: bare
# HTTP 500 "contact support" bodies for a few minutes at a time) or a 429.
# Without a retry, one blip fails the whole multi-page caller. Retry those
# statuses and connection/timeout errors with exponential backoff; a 4xx or
# an {"error": ...} envelope is not transient and still raises at once.
# Backoff between the 4 attempts: 1s, 2s, 4s.
_SEARCH_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
_SEARCH_MAX_ATTEMPTS = 4
_SEARCH_BACKOFF_BASE_S = 1.0


def _describe_unreachable(exc: requests.RequestException, attempts: int) -> str:
    """A `requests.exceptions.ConnectionError` (which is what a DNS
    resolution failure like `NameResolutionError` surfaces as) means the
    request never reached Roboflow at all — that's a local network/DNS/
    firewall/proxy problem on this machine, not "Roboflow is having a bad
    moment", and telling the user to just retry in a few minutes is wrong
    advice for it (observed live: users read that hint, wait, and hit the
    exact same DNS failure again). Anything else (timeouts, etc.) keeps the
    original "transient, retry later" framing, which is accurate for those."""
    if isinstance(exc, requests.exceptions.ConnectionError):
        return (
            f"Could not reach Roboflow (api.roboflow.com) after {attempts} attempts — this "
            "machine was unable to connect at all, which usually means a local internet, DNS, "
            f"or firewall/proxy problem rather than an issue on Roboflow's side. ({exc})"
        )
    return (
        f"Roboflow /search could not be reached after {attempts} attempts ({exc}). This is "
        "usually a temporary Roboflow-side issue — retry in a few minutes."
    )


def rf_search_page(
    rf_project,
    api_key: str,
    *,
    offset: int,
    limit: int,
    fields: list[str],
    batch_id: str | None = None,
) -> list[dict]:
    """Direct call to Roboflow's `/search` endpoint, in place of
    `rf_project.search()`. The SDK (roboflow==1.4.1) ends `search()` with a
    bare `data.json()["results"]` — no status check, no error-envelope
    check (unlike `Workspace.create_project()` right beside it, which does
    `if "error" in r.json()`) — so any response that isn't
    `{"results": [...]}` (an `{"error": ...}` body, a changed response
    shape, a permission failure on this one endpoint) surfaces only as an
    opaque `KeyError: 'results'` from deep in the SDK with the real cause
    swallowed. This issues the identical request (same URL and payload
    shape the SDK builds from `rf_project.id`) but inspects the response,
    retries a transient 5xx/429, and keeps the body in both the raised
    error and the logs.

    `batch_id`, when given, narrows results to that one upload batch —
    same as the SDK's `search(batch=True, batch_id=...)`. Note: Roboflow's
    valid `fields` list does NOT include anything that reports which batch
    an image belongs to (confirmed live: `"Invalid fields requested: ...
    Valid fields are: annotations, aspectRatio, created, embedding,
    features, filename, height, id, labels, name, owner, projects, score,
    split, tags, url, user_metadata, width, offset, limit"`) — the only way
    to learn "is image X in batch Y" is to ask for batch Y's own image list
    (via `batch_id`) and check."""
    from roboflow.config import API_URL

    url = f"{API_URL}/{rf_project.id}/search?api_key={api_key}"
    payload = {
        "offset": offset,
        "limit": limit,
        "batch": batch_id is not None,
        "fields": fields,
    }
    if batch_id is not None:
        payload["batch_id"] = batch_id

    for attempt in range(1, _SEARCH_MAX_ATTEMPTS + 1):
        try:
            resp = requests.post(url, json=payload, timeout=30)
        except requests.RequestException as exc:
            logger.warning(
                "Roboflow /search request error (attempt %d/%d): %s",
                attempt,
                _SEARCH_MAX_ATTEMPTS,
                exc,
            )
            if attempt == _SEARCH_MAX_ATTEMPTS:
                raise RuntimeError(_describe_unreachable(exc, _SEARCH_MAX_ATTEMPTS)) from exc
            time.sleep(_SEARCH_BACKOFF_BASE_S * 2 ** (attempt - 1))
            continue

        if resp.status_code in _SEARCH_RETRY_STATUSES and attempt < _SEARCH_MAX_ATTEMPTS:
            logger.warning(
                "Roboflow /search returned HTTP %s (attempt %d/%d) — retrying",
                resp.status_code,
                attempt,
                _SEARCH_MAX_ATTEMPTS,
            )
            time.sleep(_SEARCH_BACKOFF_BASE_S * 2 ** (attempt - 1))
            continue

        break

    try:
        body = resp.json()
    except ValueError as exc:
        logger.error(
            "Roboflow /search returned non-JSON (HTTP %s): %s", resp.status_code, resp.text[:2000]
        )
        raise RuntimeError(
            f"Roboflow /search returned a non-JSON response (HTTP {resp.status_code})"
        ) from exc

    if resp.status_code != 200 or not isinstance(body, dict) or "results" not in body:
        logger.error("Roboflow /search failed (HTTP %s): %s", resp.status_code, body)
        detail = (body.get("error") or body.get("message") or body) if isinstance(body, dict) else body
        hint = (
            " This is usually a temporary Roboflow-side issue — retry in a few minutes."
            if resp.status_code in _SEARCH_RETRY_STATUSES
            else ""
        )
        raise RuntimeError(f"Roboflow /search failed (HTTP {resp.status_code}): {detail}{hint}")

    return body["results"]
