"""Immich API client.

Verified against the Immich OpenAPI spec, API version 3.1.0.

Endpoints used, with the permission each one requires (the spec's
`x-immich-permission`), because a key scoped to a subset fails in ways that look
like bugs:

  GET  /server/ping                    (none)     connectivity check
  GET  /api-keys/me                    (none)     the calling key's own permissions
  GET  /search/person?name=            person.read
  GET  /people/{id}/statistics         person.statistics
  POST /search/metadata                asset.read
  GET  /faces?id={assetId}             face.read
  GET  /assets/{id}/original           asset.download
  GET  /assets/{id}/thumbnail?size=    asset.view
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any, AsyncIterator

import httpx

from .config import Credentials

JSON = "application/json"
ANY = "*/*"

_UUID = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)

# Permission required per endpoint template, used to turn a bare 403 into a
# message that names the missing scope.
ENDPOINT_PERMISSIONS = {
    "/search/person": "person.read",
    "/people/{id}/statistics": "person.statistics",
    "/search/metadata": "asset.read",
    "/faces": "face.read",
    "/assets/{id}/original": "asset.download",
    "/assets/{id}/thumbnail": "asset.view",
}

# What the pipeline needs to complete a run, and why.
REQUIRED_PERMISSIONS = {
    "person.read": "resolve the person by name",
    "asset.read": "index her photos",
    "face.read": "read face bounding boxes",
    "asset.download": "download originals",
}
OPTIONAL_PERMISSIONS = {
    "person.statistics": "detect tagging drift on old photos",
}

# Immich's Permission enum includes a real wildcard value.
WILDCARD = "all"


def normalize_path(path: str) -> str:
    """Collapse ids out of a request path so it matches an endpoint template."""
    return _UUID.sub("{id}", path)


class ImmichHTTPError(RuntimeError):
    """An HTTP error that keeps the status code.

    The predecessor of this class logged `type(exc).__name__`, which reduced
    every failure to the string "HTTPStatusError" -- indistinguishable between a
    permission problem, content negotiation and a wrong path. Keep the status,
    the path and whatever the API said.
    """

    def __init__(self, method: str, path: str, status: int, body: str = ""):
        self.method = method
        self.path = path
        self.status = status
        self.body = body
        super().__init__(self._render())

    def _render(self) -> str:
        text = f"{self.method} {self.path} -> HTTP {self.status}"
        detail = self.body.strip()
        if detail:
            text += f": {detail[:300]}"
        if self.status == 401:
            text += "\n  The API key was rejected. Check IMMICH_API_KEY."
        elif self.status == 403:
            needed = ENDPOINT_PERMISSIONS.get(normalize_path(self.path))
            if needed:
                text += (
                    f"\n  This endpoint requires the {needed!r} permission. "
                    "Check the key in Immich under Account Settings -> API Keys."
                )
        elif self.status == 406:
            text += ("\n  The server rejected the requested content type "
                     "(Accept header).")
        elif self.status == 404:
            text += ("\n  Endpoint not found -- this usually means the Immich "
                     "server is older than the API this client targets.")
        return text


def _message_from(response: httpx.Response) -> str:
    """Immich returns a JSON body with a `message` on errors; fall back to text."""
    try:
        payload = response.json()
    except ValueError:
        return response.text[:300]
    if isinstance(payload, dict):
        for key in ("message", "error", "detail"):
            value = payload.get(key)
            if value:
                return str(value)
    return str(payload)[:300]


@dataclass(frozen=True)
class Person:
    id: str
    name: str


@dataclass(frozen=True)
class Face:
    """A detected face, in the coordinate space Immich ran detection in.

    `image_width`/`image_height` are that space's dimensions -- they are NOT the
    original file's dimensions, so callers must rescale before using the box
    against a downloaded original. See `scale_bbox`.
    """

    x1: int
    y1: int
    x2: int
    y2: int
    image_width: int
    image_height: int
    source_type: str | None

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)


def missing_permissions(granted: set[str], required: dict[str, str]) -> list[str]:
    """Required permissions the key does not hold, honouring the `all` wildcard."""
    if WILDCARD in granted:
        return []
    return [name for name in required if name not in granted]


class ImmichClient:
    def __init__(self, creds: Credentials, timeout: float = 60.0, concurrency: int = 8,
                 transport: httpx.AsyncBaseTransport | None = None):
        self._client = httpx.AsyncClient(
            base_url=creds.url,
            # Accept is set per request: the download endpoints produce
            # application/octet-stream, so a client-wide JSON default would be a
            # lie on exactly the requests that matter most.
            headers={"x-api-key": creds.api_key},
            timeout=timeout,
            follow_redirects=True,
            transport=transport,
        )
        self._sem = asyncio.Semaphore(concurrency)

    async def __aenter__(self) -> "ImmichClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._client.aclose()

    def _check(self, response: httpx.Response, method: str, path: str) -> httpx.Response:
        if response.is_success:
            return response
        raise ImmichHTTPError(method, path, response.status_code, _message_from(response))

    async def _get(self, path: str, accept: str = JSON, **kwargs: Any) -> httpx.Response:
        headers = {"Accept": accept, **kwargs.pop("headers", {})}
        async with self._sem:
            response = await self._client.get(path, headers=headers, **kwargs)
        return self._check(response, "GET", path)

    async def _post(self, path: str, json: Any) -> httpx.Response:
        async with self._sem:
            response = await self._client.post(path, json=json, headers={"Accept": JSON})
        return self._check(response, "POST", path)

    async def ping(self) -> None:
        """Fail fast and clearly on a bad URL or key, before any long stage."""
        await self._get("/server/ping")

    async def probe(self, path: str, accept: str = JSON,
                    params: dict | None = None) -> dict[str, Any]:
        """Make one request and describe the response without raising.

        For diagnostics: when a stage fails on every item, the useful question is
        what a single request actually returns, headers and all.
        """
        try:
            async with self._sem:
                response = await self._client.get(path, headers={"Accept": accept},
                                                  params=params)
        except httpx.HTTPError as exc:
            return {"path": path, "ok": False, "status": None,
                    "detail": f"{type(exc).__name__}: {exc}"}

        return {
            "path": path,
            "ok": response.is_success,
            "status": response.status_code,
            "content_type": response.headers.get("content-type", ""),
            "length": len(response.content),
            "detail": "" if response.is_success else _message_from(response),
        }

    async def my_permissions(self) -> set[str]:
        """The calling key's own permissions.

        `/api-keys/me` requires no permission at all, so this works even for a
        key too narrow to do anything else -- which is what makes it usable as a
        preflight.
        """
        payload = (await self._get("/api-keys/me")).json()
        return {str(p) for p in payload.get("permissions", [])}

    async def resolve_person(self, name: str) -> Person:
        people = (await self._get("/search/person",
                                  params={"name": name, "withHidden": "false"})).json()
        exact = [p for p in people if (p.get("name") or "").strip().lower() == name.strip().lower()]
        candidates = exact or people
        if not candidates:
            raise RuntimeError(f"no person named {name!r} found in Immich")
        if len(candidates) > 1:
            listing = ", ".join(f"{p['name']} ({p['id']})" for p in candidates[:5])
            raise RuntimeError(
                f"{len(candidates)} people match {name!r}: {listing}. "
                "Set immich.person_id in config.toml to disambiguate."
            )
        p = candidates[0]
        return Person(id=p["id"], name=p.get("name") or "")

    async def person_asset_count(self, person_id: str) -> int | None:
        """Assets this person appears in, or None if the count is unavailable.

        Used as the drift detector for the sync watermark: tagging a person in an
        *old* photo need not bump that asset's `updatedAt`, so an `updatedAfter`
        query can miss it. A jump in this count reveals that.

        Returning None degrades to "assume no drift", which is safe but silent --
        callers should say so out loud.
        """
        try:
            resp = await self._get(f"/people/{person_id}/statistics")
        except ImmichHTTPError:
            return None
        return int(resp.json().get("assets", 0))

    async def search_assets(
        self,
        person_id: str,
        updated_after: str | None = None,
        page_size: int = 1000,
    ) -> AsyncIterator[dict]:
        """Yield every IMAGE asset the person appears in.

        Pagination follows `nextPage`; `total` is deprecated in API v3 and is
        not relied on.
        """
        page = 1
        while True:
            body: dict[str, Any] = {
                "personIds": [person_id],
                "type": "IMAGE",
                "size": page_size,
                "page": page,
                "withExif": True,
                "order": "asc",
            }
            if updated_after:
                body["updatedAfter"] = updated_after

            assets = (await self._post("/search/metadata", body)).json()["assets"]
            for item in assets.get("items", []):
                yield item

            next_page = assets.get("nextPage")
            if not next_page:
                return
            # nextPage is a token; when it is numeric, honour it rather than
            # assuming a simple increment.
            page = int(next_page) if str(next_page).isdigit() else page + 1

    async def faces_for_asset(self, asset_id: str) -> list[dict]:
        return (await self._get("/faces", params={"id": asset_id})).json()

    async def download(self, asset_id: str, source: str = "original") -> bytes:
        """Fetch file bytes.

        Accept is `*/*` here: these endpoints produce application/octet-stream,
        and asking for JSON invites a 406 on the one request that has to return
        binary.
        """
        if source == "original":
            resp = await self._get(f"/assets/{asset_id}/original", accept=ANY)
        else:
            resp = await self._get(f"/assets/{asset_id}/thumbnail",
                                   accept=ANY, params={"size": source})
        return resp.content


def pick_face(faces: list[dict], person_id: str) -> tuple[Face | None, int]:
    """Choose the target person's face box from an asset's detected faces.

    Returns (face, n_candidates). Immich associates each detected face with a
    person, so group photos need no recognition on our side. Where a merge has
    left more than one face on the same person, the largest wins -- and the
    count is returned so the caller can flag it.
    """
    matches = [
        f for f in faces
        if isinstance(f.get("person"), dict) and f["person"].get("id") == person_id
    ]
    if not matches:
        return None, 0
    best = max(matches, key=lambda f: (f["boundingBoxX2"] - f["boundingBoxX1"])
               * (f["boundingBoxY2"] - f["boundingBoxY1"]))
    face = Face(
        x1=int(best["boundingBoxX1"]),
        y1=int(best["boundingBoxY1"]),
        x2=int(best["boundingBoxX2"]),
        y2=int(best["boundingBoxY2"]),
        image_width=int(best["imageWidth"]),
        image_height=int(best["imageHeight"]),
        source_type=best.get("sourceType"),
    )
    return face, len(matches)


class AspectMismatch(ValueError):
    """Raised when a face box cannot be mapped onto the downloaded image."""


def scale_bbox(face: Face, target_width: int, target_height: int,
               tolerance: float = 0.02) -> tuple[int, int, int, int]:
    """Map a face box from Immich's detection space onto a decoded image.

    Immich reports boxes against a rotation-applied rendition, so the decoded
    image must also be EXIF-oriented before this is called. A mismatched aspect
    ratio means the two are in different orientations and the box would land on
    the wrong part of the face -- loudly refuse rather than silently mis-crop.
    """
    if face.image_width <= 0 or face.image_height <= 0:
        raise AspectMismatch("face record has no image dimensions")

    src_aspect = face.image_width / face.image_height
    dst_aspect = target_width / target_height
    if abs(src_aspect - dst_aspect) / src_aspect > tolerance:
        raise AspectMismatch(
            f"aspect mismatch: detection space {face.image_width}x{face.image_height} "
            f"({src_aspect:.3f}) vs image {target_width}x{target_height} ({dst_aspect:.3f})"
        )

    sx = target_width / face.image_width
    sy = target_height / face.image_height
    return (
        int(round(face.x1 * sx)),
        int(round(face.y1 * sy)),
        int(round(face.x2 * sx)),
        int(round(face.y2 * sy)),
    )
