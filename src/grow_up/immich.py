"""Immich API client.

Verified against the Immich OpenAPI spec, API version 3.1.0.

Endpoints used:
  GET  /search/person?name=            resolve a person by name
  GET  /people/{id}/statistics         authoritative count of assets a person appears in
  POST /search/metadata                paginated asset search, filtered by personIds
  GET  /faces?id={assetId}             every detected face on an asset, with person + bbox
  GET  /assets/{id}/original           the original file
  GET  /assets/{id}/thumbnail?size=    preview/fullsize renditions
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, AsyncIterator

import httpx

from .config import Credentials


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


class ImmichClient:
    def __init__(self, creds: Credentials, timeout: float = 60.0, concurrency: int = 8):
        self._client = httpx.AsyncClient(
            base_url=creds.url,
            headers={"x-api-key": creds.api_key, "Accept": "application/json"},
            timeout=timeout,
            follow_redirects=True,
        )
        self._sem = asyncio.Semaphore(concurrency)

    async def __aenter__(self) -> "ImmichClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._client.aclose()

    async def _get(self, url: str, **kwargs: Any) -> httpx.Response:
        async with self._sem:
            resp = await self._client.get(url, **kwargs)
        resp.raise_for_status()
        return resp

    async def ping(self) -> None:
        """Fail fast and clearly on a bad URL or key, before any long stage."""
        resp = await self._client.get("/server/ping")
        if resp.status_code == 401:
            raise RuntimeError("Immich rejected the API key (401). Check IMMICH_API_KEY.")
        resp.raise_for_status()

    async def resolve_person(self, name: str) -> Person:
        resp = await self._get("/search/person", params={"name": name, "withHidden": "false"})
        people = resp.json()
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
        """Assets this person appears in.

        Used as the drift detector for the sync watermark: tagging a person in an
        *old* photo need not bump that asset's `updatedAt`, so an `updatedAfter`
        query can miss it. A jump in this count reveals that.
        """
        try:
            resp = await self._get(f"/people/{person_id}/statistics")
        except httpx.HTTPStatusError:
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

            async with self._sem:
                resp = await self._client.post("/search/metadata", json=body)
            resp.raise_for_status()
            assets = resp.json()["assets"]

            for item in assets.get("items", []):
                yield item

            next_page = assets.get("nextPage")
            if not next_page:
                return
            # nextPage is a token; when it is numeric, honour it rather than
            # assuming a simple increment.
            page = int(next_page) if str(next_page).isdigit() else page + 1

    async def faces_for_asset(self, asset_id: str) -> list[dict]:
        resp = await self._get("/faces", params={"id": asset_id})
        return resp.json()

    async def download(self, asset_id: str, source: str = "original") -> bytes:
        if source == "original":
            resp = await self._get(f"/assets/{asset_id}/original")
        else:
            resp = await self._get(f"/assets/{asset_id}/thumbnail", params={"size": source})
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
