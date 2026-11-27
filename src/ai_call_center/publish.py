"""Publish the public report as one JSON file in a Cloudflare R2 bucket.

The page that shows the report is on another host. It reads this file, so a visitor to the page never
starts this server. The file is `analytics.public_report`: figures and verdicts, with no call id, no
grader note, and nothing a caller said.

The upload is one signed PUT to R2's S3 API. The signing is AWS Signature Version 4, written here with
the standard library, so the project needs no AWS package for one request. The four REPORT_R2 settings
turn the upload on; with one of them empty there is no upload.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import quote

from ai_call_center.analytics import AnalyticsStore, dated_report
from ai_call_center.config import Settings
from ai_call_center.fix_check import stored_checks

log = logging.getLogger(__name__)

R2_REGION = "auto"  # R2 has one region name for every bucket
SEND_TIME_LIMIT = 10.0  # seconds; the call is over, so nobody waits for this
CACHE_CONTROL = "public, max-age=60"


@dataclass(frozen=True)
class Target:
    account_id: str
    bucket: str
    key: str
    access_key_id: str
    secret_access_key: str

    @property
    def host(self) -> str:
        return f"{self.account_id}.r2.cloudflarestorage.com"

    @property
    def path(self) -> str:
        return "/" + quote(f"{self.bucket}/{self.key}", safe="/-_.~")


def target(settings: Settings) -> Target | None:
    """Where the report goes, or None when the upload is not set up."""
    made = Target(
        account_id=settings.report_r2_account_id.strip(),
        bucket=settings.report_r2_bucket.strip(),
        key=settings.report_r2_key.strip().lstrip("/"),
        access_key_id=settings.report_r2_access_key_id.strip(),
        secret_access_key=settings.report_r2_secret_access_key.get_secret_value().strip(),
    )
    return made if all((made.account_id, made.bucket, made.key, made.access_key_id, made.secret_access_key)) else None


def _hmac(key: bytes, text: str) -> bytes:
    return hmac.new(key, text.encode(), hashlib.sha256).digest()


def authorization(method: str, path: str, headers: dict[str, str], *, access_key_id: str, secret_access_key: str, region: str, service: str = "s3") -> str:
    """The `Authorization` header of AWS Signature Version 4, for a request with no query string.

    `headers` are the headers to sign. They must hold `host`, `x-amz-date` and `x-amz-content-sha256`."""
    signed = {name.lower(): " ".join(value.split()) for name, value in headers.items()}
    names = ";".join(sorted(signed))
    stamp = signed["x-amz-date"]
    scope = f"{stamp[:8]}/{region}/{service}/aws4_request"
    request = "\n".join([method, path, "", *(f"{name}:{signed[name]}" for name in sorted(signed)), "", names, signed["x-amz-content-sha256"]])
    to_sign = "\n".join(["AWS4-HMAC-SHA256", stamp, scope, hashlib.sha256(request.encode()).hexdigest()])
    key = f"AWS4{secret_access_key}".encode()
    for part in (stamp[:8], region, service, "aws4_request"):
        key = _hmac(key, part)
    signature = hmac.new(key, to_sign.encode(), hashlib.sha256).hexdigest()
    return f"AWS4-HMAC-SHA256 Credential={access_key_id}/{scope},SignedHeaders={names},Signature={signature}"


def put_request(to: Target, body: bytes, *, now: datetime | None = None) -> urllib.request.Request:
    """The signed request that writes `body` to the target."""
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    signed = {"host": to.host, "x-amz-content-sha256": hashlib.sha256(body).hexdigest(), "x-amz-date": stamp}
    auth = authorization("PUT", to.path, signed, access_key_id=to.access_key_id, secret_access_key=to.secret_access_key, region=R2_REGION)
    headers = {**signed, "Authorization": auth, "Content-Type": "application/json", "Cache-Control": CACHE_CONTROL}
    del headers["host"]  # urllib sets it from the URL
    return urllib.request.Request(f"https://{to.host}{to.path}", data=body, method="PUT", headers=headers)


def _send(request: urllib.request.Request) -> int:
    with urllib.request.urlopen(request, timeout=SEND_TIME_LIMIT) as response:  # noqa: S310 - the URL is always https, made in put_request
        return response.status


def report_body(settings: Settings) -> bytes:
    """The public report of the calls in the store, as JSON. Test calls are left out unless `report_tests` is set."""
    store = AnalyticsStore(settings.leads_db)
    try:
        calls = store.calls()
    finally:
        store.close()
    report = dated_report(calls, tests=settings.report_tests, fix_checks=stored_checks(settings.leads_db))
    return json.dumps(report, separators=(",", ":")).encode()


async def publish_report(settings: Settings, *, send: Callable[[urllib.request.Request], int] = _send) -> bool:
    """Write the public report to the bucket. True when the file was written.

    It never raises: the end of a call must not fail because a file did not upload. A failure is one log line."""
    to = target(settings)
    if to is None:
        return False
    try:
        body = report_body(settings)
        status = await asyncio.to_thread(send, put_request(to, body))
    except Exception as error:
        # The error of urllib holds the status and its reason. It does not hold the key.
        log.warning("the public report did not upload to %s/%s: %s", to.bucket, to.key, error)
        return False
    log.info("the public report is in %s/%s (%d bytes, HTTP %s)", to.bucket, to.key, len(body), status)
    return True
