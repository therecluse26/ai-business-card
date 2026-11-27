"""The upload of the public report: the signature, the request, and what the file holds."""

import hashlib
import json
import urllib.error
from datetime import UTC, datetime

from pydantic import SecretStr

from ai_call_center.analytics import AnalyticsStore
from ai_call_center.config import Settings
from ai_call_center.publish import authorization, publish_report, put_request, target
from tests.test_analytics import CALL_ID, SECRET_NOTE, a_call

R2 = {"report_r2_account_id": "acct123", "report_r2_bucket": "voice-card", "report_r2_access_key_id": "key-id", "report_r2_secret_access_key": SecretStr("key-secret")}


def settings(tmp_path, **over) -> Settings:
    return Settings(_env_file=None, **{"leads_db": tmp_path / "leads.sqlite3", **R2, **over})


def test_the_signature_is_the_one_in_the_aws_documentation():
    # "Example: GET Object" in the Amazon S3 reference for Signature Version 4. The keys are AWS's public examples.
    headers = {
        "Host": "examplebucket.s3.amazonaws.com",
        "Range": "bytes=0-9",
        "x-amz-content-sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "x-amz-date": "20130524T000000Z",
    }
    got = authorization("GET", "/test.txt", headers, access_key_id="AKIAIOSFODNN7EXAMPLE", secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", region="us-east-1")
    assert got == (
        "AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/20130524/us-east-1/s3/aws4_request,"
        "SignedHeaders=host;range;x-amz-content-sha256;x-amz-date,"
        "Signature=f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41"
    )


def test_with_one_setting_empty_there_is_no_target(tmp_path):
    assert target(settings(tmp_path)) is not None
    for name in ("report_r2_account_id", "report_r2_bucket", "report_r2_access_key_id", "report_r2_key"):
        assert target(settings(tmp_path, **{name: " "})) is None
    assert target(settings(tmp_path, report_r2_secret_access_key=SecretStr(""))) is None


def test_the_request_is_a_signed_put_of_json_to_the_bucket(tmp_path):
    request = put_request(target(settings(tmp_path, report_r2_key="/site/calls.json")), b'{"calls":[]}', now=datetime(2026, 11, 21, 5, 0, 7, tzinfo=UTC))
    assert (request.get_method(), request.full_url) == ("PUT", "https://acct123.r2.cloudflarestorage.com/voice-card/site/calls.json")
    headers = {name.lower(): value for name, value in request.header_items()}
    assert headers["content-type"] == "application/json" and headers["x-amz-date"] == "20261121T050007Z"
    assert headers["x-amz-content-sha256"] == hashlib.sha256(b'{"calls":[]}').hexdigest()
    assert headers["authorization"].startswith("AWS4-HMAC-SHA256 Credential=key-id/20261121/auto/s3/aws4_request,SignedHeaders=host;x-amz-content-sha256;x-amz-date,Signature=")
    assert "key-secret" not in str(request.header_items()) and request.data == b'{"calls":[]}'


async def test_the_file_is_the_public_report_with_no_test_call_and_nothing_private(tmp_path):
    made = settings(tmp_path)
    store = AnalyticsStore(made.leads_db)
    store.save(a_call())
    store.save(a_call("mine", test=True))
    store.close()
    sent = []
    assert await publish_report(made, send=lambda request: sent.append(request) or 200) is True
    [request] = sent
    report = json.loads(request.data)
    assert report["totals"]["calls"] == 1 and len(report["calls"]) == 1 and report["generated_on"]
    assert report["totals"]["test_calls"] == 0
    assert report["fix_checks"] == []  # no fix was checked yet; the field is there all the same
    text = request.data.decode()
    assert CALL_ID not in text and SECRET_NOTE not in text and "mine" not in text


async def test_with_report_tests_the_file_holds_the_test_calls_and_counts_them(tmp_path):
    made = settings(tmp_path, report_tests=True)
    store = AnalyticsStore(made.leads_db)
    store.save(a_call())
    store.save(a_call("mine", test=True))
    store.close()
    sent = []
    assert await publish_report(made, send=lambda request: sent.append(request) or 200) is True
    report = json.loads(sent[0].data)
    assert report["totals"]["calls"] == 2 and report["totals"]["test_calls"] == 1 and len(report["calls"]) == 2
    text = sent[0].data.decode()
    assert CALL_ID not in text and SECRET_NOTE not in text and "mine" not in text


async def test_with_no_target_nothing_is_sent(tmp_path):
    sent = []
    assert await publish_report(settings(tmp_path, report_r2_bucket=""), send=sent.append) is False and sent == []


async def test_a_failed_upload_is_false_and_never_an_error(tmp_path):
    def refused(request):
        raise urllib.error.HTTPError(request.full_url, 403, "Forbidden", None, None)  # type: ignore[arg-type]

    assert await publish_report(settings(tmp_path), send=refused) is False
