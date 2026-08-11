"""HTTP error reporting, content negotiation and the permission preflight.

Regression cover for a real failure: every download against a live library died
with the log line `HTTPStatusError` and nothing else, which is indistinguishable
between a permission problem, content negotiation and a wrong path.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from grow_up import cli, pipeline
from grow_up.config import Credentials
from grow_up.immich import (
    ANY,
    JSON,
    REQUIRED_PERMISSIONS,
    ImmichClient,
    ImmichHTTPError,
    missing_permissions,
    normalize_path,
)

CREDS = Credentials(url="https://immich.example.com/api", api_key="secret")
ASSET = "03529b3b-e721-40d1-a8a2-9443398e6f69"


def client_with(handler, retries: int = 0, sleeps: list | None = None) -> ImmichClient:
    """A client wired to a mock transport.

    Retries default to 0 and sleeping is captured rather than performed, so the
    suite stays fast and the backoff schedule is assertable.
    """
    async def sleeper(seconds: float) -> None:
        if sleeps is not None:
            sleeps.append(seconds)

    return ImmichClient(CREDS, transport=httpx.MockTransport(handler),
                        retries=retries, sleeper=sleeper)


def run(coro):
    return asyncio.run(coro)


class TestErrorRendering:
    def test_keeps_the_status_code(self):
        exc = ImmichHTTPError("GET", f"/assets/{ASSET}/original", 403, "Forbidden")
        assert "403" in str(exc)
        assert f"/assets/{ASSET}/original" in str(exc)
        assert "Forbidden" in str(exc)

    def test_403_names_the_permission_the_endpoint_needs(self):
        exc = ImmichHTTPError("GET", f"/assets/{ASSET}/original", 403, "")
        assert "asset.download" in str(exc)

    def test_403_on_a_different_endpoint_names_its_own_permission(self):
        assert "face.read" in str(ImmichHTTPError("GET", "/faces", 403, ""))
        assert "asset.read" in str(ImmichHTTPError("POST", "/search/metadata", 403, ""))

    def test_401_points_at_the_key(self):
        assert "IMMICH_API_KEY" in str(ImmichHTTPError("GET", "/server/ping", 401, ""))

    def test_406_points_at_content_negotiation(self):
        assert "Accept" in str(ImmichHTTPError("GET", f"/assets/{ASSET}/original", 406, ""))

    def test_404_points_at_a_version_mismatch(self):
        assert "older" in str(ImmichHTTPError("GET", f"/assets/{ASSET}/original", 404, ""))

    def test_normalize_path_collapses_uuids(self):
        assert normalize_path(f"/assets/{ASSET}/original") == "/assets/{id}/original"
        assert normalize_path("/faces") == "/faces"


class TestErrorsFromResponses:
    def test_surfaces_the_api_message(self):
        def handler(request):
            return httpx.Response(403, json={"message": "Not permitted", "statusCode": 403})

        with pytest.raises(ImmichHTTPError) as excinfo:
            run(client_with(handler).download(ASSET))

        assert excinfo.value.status == 403
        assert "Not permitted" in str(excinfo.value)

    def test_survives_a_non_json_error_body(self):
        def handler(request):
            return httpx.Response(502, text="<html>bad gateway</html>")

        with pytest.raises(ImmichHTTPError) as excinfo:
            run(client_with(handler).download(ASSET))

        assert excinfo.value.status == 502

    def test_success_returns_bytes(self):
        def handler(request):
            return httpx.Response(200, content=b"\xff\xd8\xff-jpeg-bytes")

        assert run(client_with(handler).download(ASSET)) == b"\xff\xd8\xff-jpeg-bytes"


class TestContentNegotiation:
    """The client used to send `Accept: application/json` on every request,
    including the endpoints that only ever produce application/octet-stream."""

    def capture(self):
        seen = {}

        def handler(request):
            seen[str(request.url.path)] = request.headers.get("accept")
            if "original" in str(request.url) or "thumbnail" in str(request.url):
                return httpx.Response(200, content=b"bytes")
            return httpx.Response(200, json=[])

        return seen, handler

    def test_download_asks_for_anything(self):
        seen, handler = self.capture()
        run(client_with(handler).download(ASSET))
        assert seen[f"/api/assets/{ASSET}/original"] == ANY

    def test_thumbnail_asks_for_anything(self):
        seen, handler = self.capture()
        run(client_with(handler).download(ASSET, "preview"))
        assert seen[f"/api/assets/{ASSET}/thumbnail"] == ANY

    def test_json_endpoints_still_ask_for_json(self):
        seen, handler = self.capture()
        run(client_with(handler).faces_for_asset(ASSET))
        assert seen["/api/faces"] == JSON

    def test_api_key_header_is_still_sent(self):
        sent = {}

        def handler(request):
            sent["key"] = request.headers.get("x-api-key")
            return httpx.Response(200, content=b"bytes")

        run(client_with(handler).download(ASSET))
        assert sent["key"] == "secret"


class TestPermissionPreflight:
    def test_exact_permissions_are_enough(self):
        assert missing_permissions(set(REQUIRED_PERMISSIONS), REQUIRED_PERMISSIONS) == []

    def test_reports_only_what_is_missing(self):
        granted = set(REQUIRED_PERMISSIONS) - {"asset.download"}
        assert missing_permissions(granted, REQUIRED_PERMISSIONS) == ["asset.download"]

    def test_all_is_a_real_wildcard(self):
        """Immich's Permission enum contains a literal `all` value."""
        assert missing_permissions({"all"}, REQUIRED_PERMISSIONS) == []

    def test_empty_key_reports_everything(self):
        assert set(missing_permissions(set(), REQUIRED_PERMISSIONS)) == set(REQUIRED_PERMISSIONS)

    def test_my_permissions_reads_the_key(self):
        def handler(request):
            assert request.url.path == "/api/api-keys/me"
            return httpx.Response(200, json={"id": "x", "name": "k",
                                             "permissions": ["all"]})

        assert run(client_with(handler).my_permissions()) == {"all"}


def key_with(*permissions: str):
    """A transport where ping succeeds and the key holds exactly these scopes."""
    def handler(request):
        if request.url.path.endswith("/server/ping"):
            return httpx.Response(200, json={"res": "pong"})
        return httpx.Response(200, json={"id": "x", "name": "k",
                                         "permissions": list(permissions)})
    return handler


class TestWhatPreflightSays:
    """A healthy preflight must stay silent, and an unhealthy one must name the key.

    Both were found by a real two-account run: `preflight_all` printed a heading
    per source for output that, on good keys, never came -- and the message for a
    missing required permission said "this Immich API key" without saying which.
    """

    @pytest.fixture()
    def said(self, monkeypatch):
        lines: list[str] = []
        monkeypatch.setattr(cli, "log", lines.append)
        return lines

    def test_a_healthy_key_says_nothing_at_all(self, said):
        """The root of the empty-heading bug: there is nothing to head."""
        run(cli.preflight(client_with(key_with("all"))))
        assert said == []

    def test_a_healthy_key_says_nothing_when_named_either(self, said):
        run(cli.preflight(client_with(key_with("all")), " for source 'her'"))
        assert said == []

    def test_a_missing_optional_permission_is_a_note(self, said):
        run(cli.preflight(client_with(key_with(*REQUIRED_PERMISSIONS))))
        assert len(said) == 1
        assert "person.statistics" in said[0]

    def test_that_note_names_the_account(self, said):
        run(cli.preflight(client_with(key_with(*REQUIRED_PERMISSIONS)),
                          " for source 'her'"))
        assert "'her'" in said[0]

    def test_a_missing_required_permission_names_the_account(self):
        granted = set(REQUIRED_PERMISSIONS) - {"asset.download"}
        with pytest.raises(RuntimeError, match="for source 'her'"):
            run(cli.preflight(client_with(key_with(*granted)), " for source 'her'"))

    def test_an_unreadable_key_names_the_account(self, said):
        def handler(request):
            if request.url.path.endswith("/server/ping"):
                return httpx.Response(200, json={"res": "pong"})
            return httpx.Response(403, json={"message": "nope"})

        assert run(cli.preflight(client_with(handler), " for source 'her'")) == set()
        assert "'her'" in said[0] and "403" in said[0]

    def test_one_account_keeps_the_wording_it_always_had(self, said):
        """Pinned: a single-account run's strings must not drift."""
        granted = set(REQUIRED_PERMISSIONS) - {"asset.download"}
        with pytest.raises(RuntimeError) as caught:
            run(cli.preflight(client_with(key_with(*granted))))
        assert str(caught.value).startswith(
            "this Immich API key is missing required permission(s):")

        # Only the wording up to the reason: the reason itself lives in
        # OPTIONAL_PERMISSIONS and duplicating it here would pin the wrong thing.
        run(cli.preflight(client_with(key_with(*REQUIRED_PERMISSIONS))))
        assert said[0].startswith("  note: the key lacks 'person.statistics', so ")


class TestProbe:
    def test_describes_a_failure_without_raising(self):
        def handler(request):
            return httpx.Response(403, json={"message": "nope"})

        result = run(client_with(handler).probe(f"/assets/{ASSET}/original", ANY))
        assert result["ok"] is False
        assert result["status"] == 403
        assert "nope" in result["detail"]

    def test_describes_a_success(self):
        def handler(request):
            return httpx.Response(200, content=b"1234",
                                  headers={"content-type": "application/octet-stream"})

        result = run(client_with(handler).probe(f"/assets/{ASSET}/original", ANY))
        assert result["ok"] is True
        assert result["status"] == 200
        assert result["length"] == 4
        assert result["content_type"] == "application/octet-stream"

    def test_transport_errors_do_not_raise(self):
        def handler(request):
            raise httpx.ConnectError("no route to host")

        result = run(client_with(handler).probe("/server/ping"))
        assert result["ok"] is False and result["status"] is None
        assert "ConnectError" in result["detail"]


class TestAbortIfHopeless:
    """A stage that achieves nothing must not report success and let `run`
    continue into analyze, where the failure resurfaces as an empty video."""

    def test_total_failure_raises_with_the_first_real_error(self):
        with pytest.raises(pipeline.StageFailed, match="asset.download"):
            pipeline._abort_if_hopeless(
                "download", 0,
                ["GET /assets/x/original -> HTTP 403: needs asset.download"], 832
            )

    def test_widespread_failure_raises(self):
        with pytest.raises(pipeline.StageFailed, match="200 of 832"):
            pipeline._abort_if_hopeless("download", 632, ["boom"] * 200, 832)

    def test_a_few_failures_are_tolerated(self):
        pipeline._abort_if_hopeless("download", 830, ["boom"] * 2, 832)

    def test_no_failures_is_a_no_op(self):
        pipeline._abort_if_hopeless("download", 832, [], 832)


class TestErrorLogging:
    def test_logs_the_error_itself_not_its_class_name(self):
        lines: list[str] = []
        exc = ImmichHTTPError("GET", f"/assets/{ASSET}/original", 403, "Forbidden")
        pipeline._log_error(lines.append, ["e"], "download x", exc)

        assert "403" in lines[0]
        assert "HTTPStatusError" not in lines[0]

    def test_suppresses_after_a_handful(self):
        lines: list[str] = []
        errors: list[str] = []
        for i in range(50):
            errors.append("boom")
            pipeline._log_error(lines.append, errors, f"download {i}", RuntimeError("boom"))

        assert len(lines) == pipeline.MAX_LOGGED_ERRORS + 1
        assert "suppressed" in lines[-1]


class TestRetry:
    """Bulk-fetching hundreds of large originals is the workload that trips rate
    limiters and provokes momentary 502s; one blip should not cost the asset."""

    def counting_handler(self, statuses):
        calls = {"n": 0}

        def handler(request):
            index = min(calls["n"], len(statuses) - 1)
            calls["n"] += 1
            status = statuses[index]
            if status == 200:
                return httpx.Response(200, content=b"jpeg")
            return httpx.Response(status, json={"message": "busy"})

        return calls, handler

    def test_recovers_from_a_transient_failure(self):
        calls, handler = self.counting_handler([503, 503, 200])
        assert run(client_with(handler, retries=4).download(ASSET)) == b"jpeg"
        assert calls["n"] == 3

    def test_gives_up_after_the_retry_budget(self):
        calls, handler = self.counting_handler([503])
        with pytest.raises(ImmichHTTPError) as excinfo:
            run(client_with(handler, retries=2).download(ASSET))
        assert excinfo.value.status == 503
        assert calls["n"] == 3, "initial attempt plus two retries"

    @pytest.mark.parametrize("status", [401, 403, 404, 400])
    def test_permanent_failures_are_not_retried(self, status):
        """Retrying a permission error only delays the real message."""
        calls, handler = self.counting_handler([status])
        with pytest.raises(ImmichHTTPError):
            run(client_with(handler, retries=4).download(ASSET))
        assert calls["n"] == 1

    def test_backoff_grows(self):
        sleeps: list[float] = []
        _, handler = self.counting_handler([503])
        with pytest.raises(ImmichHTTPError):
            run(client_with(handler, retries=3, sleeps=sleeps).download(ASSET))
        assert sleeps == [1.0, 2.0, 4.0]

    def test_retry_after_header_is_honoured(self):
        def handler(request):
            return httpx.Response(429, headers={"retry-after": "7"}, json={"message": "slow"})

        sleeps: list[float] = []
        with pytest.raises(ImmichHTTPError):
            run(client_with(handler, retries=1, sleeps=sleeps).download(ASSET))
        assert sleeps == [7.0]

    def test_transport_errors_are_retried_then_reported(self):
        def handler(request):
            raise httpx.ConnectError("connection reset")

        with pytest.raises(ImmichHTTPError, match="ConnectError"):
            run(client_with(handler, retries=2).download(ASSET))

    def test_delay_is_capped(self):
        from grow_up.immich import retry_delay

        assert retry_delay(20) == 30.0
        assert retry_delay(0, retry_after=900) == 30.0

    def test_retry_after_parsing(self):
        from grow_up.immich import parse_retry_after

        assert parse_retry_after("12") == 12.0
        assert parse_retry_after(None) is None
        assert parse_retry_after("Wed, 21 Oct 2026 07:28:00 GMT") is None


class TestStreamingDownload:
    def test_writes_the_file(self, tmp_path):
        def handler(request):
            return httpx.Response(200, content=b"a" * 5000)

        target = tmp_path / "asset.jpg"
        written = run(client_with(handler).download_to(ASSET, target))

        assert written == 5000
        assert target.read_bytes() == b"a" * 5000

    def test_leaves_no_partial_file_on_failure(self, tmp_path):
        """The fetch stage treats an existing file as cached, so a truncated one
        would never be re-fetched and would fail much later, in decoding."""
        def handler(request):
            return httpx.Response(500, json={"message": "boom"})

        target = tmp_path / "asset.jpg"
        with pytest.raises(ImmichHTTPError):
            run(client_with(handler).download_to(ASSET, target))

        assert not target.exists()
        assert list(tmp_path.iterdir()) == [], "no .part file left behind"

    def test_creates_the_cache_directory(self, tmp_path):
        def handler(request):
            return httpx.Response(200, content=b"x")

        target = tmp_path / "nested" / "dir" / "asset.jpg"
        run(client_with(handler).download_to(ASSET, target))
        assert target.exists()

    def test_uses_the_thumbnail_endpoint_for_previews(self, tmp_path):
        seen = {}

        def handler(request):
            seen["path"] = request.url.path
            seen["size"] = request.url.params.get("size")
            return httpx.Response(200, content=b"x")

        run(client_with(handler).download_to(ASSET, tmp_path / "a.jpg", "preview"))
        assert seen["path"] == f"/api/assets/{ASSET}/thumbnail"
        assert seen["size"] == "preview"


def test_search_assets_reports_errors_with_status():
    def handler(request):
        return httpx.Response(403, json={"message": "no asset.read"})

    async def go():
        async for _ in client_with(handler).search_assets("person-1"):
            pass

    with pytest.raises(ImmichHTTPError) as excinfo:
        run(go())
    assert excinfo.value.status == 403
    assert excinfo.value.method == "POST"


def test_search_assets_paginates():
    pages = {
        1: {"assets": {"items": [{"id": "a"}], "nextPage": "2"}},
        2: {"assets": {"items": [{"id": "b"}], "nextPage": None}},
    }

    def handler(request):
        page = json.loads(request.content)["page"]
        return httpx.Response(200, json=pages[page])

    async def go():
        return [item async for item in client_with(handler).search_assets("person-1")]

    assert [i["id"] for i in run(go())] == ["a", "b"]
