"""Caller lookup: started early, cached, and never able to break or delay a call."""

import asyncio

from ai_call_center.callers import lookup_text
from ai_call_center.leads import NumberInfo
from ai_call_center.telephony.lookup import CallerLookup

NUMBER = "+15555550123"
INFO = NumberInfo(caller_name="STEVE PARK", carrier="Verizon Wireless", line_type="mobile", city="AKRON", state="OH")


class FakeLookup:
    def __init__(self, info=INFO, delay=0.0, fail=False):
        self.info, self.delay, self.fail, self.calls = info, delay, fail, []

    async def lookup_number(self, number, *, with_name):
        self.calls.append((number, with_name))
        await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError("lookup down")
        return self.info


async def test_lookup_is_cached_so_a_number_costs_one_lookup(leads):
    client = FakeLookup()
    lookups = CallerLookup(client, leads)
    lookups.start(NUMBER)
    assert await lookups.get(NUMBER) == INFO
    lookups.start(NUMBER)  # the second call from this number
    assert await lookups.get(NUMBER) == INFO
    assert client.calls == [(NUMBER, True)]


async def test_a_slow_lookup_does_not_delay_the_call(leads):
    lookups = CallerLookup(FakeLookup(delay=0.5), leads)
    lookups.start(NUMBER)
    assert await lookups.get(NUMBER, timeout=0.02) is None
    await asyncio.sleep(0.6)  # it still lands in the cache for the next call
    assert leads.number_info_for(NUMBER) == INFO


async def test_a_failed_lookup_is_swallowed(leads):
    lookups = CallerLookup(FakeLookup(fail=True), leads)
    lookups.start(NUMBER)
    assert await lookups.get(NUMBER) is None


async def test_off_and_hidden_numbers_are_never_looked_up(leads):
    client = FakeLookup()
    off = CallerLookup(client, leads, mode="off")
    off.start(NUMBER)
    on = CallerLookup(client, leads)
    on.start("anonymous")
    assert await off.get(NUMBER) is None and await on.get("anonymous") is None
    assert client.calls == []


def test_lookup_text_marks_the_name_as_a_guess():
    text = lookup_text(INFO)
    assert "Registered name: Steve Park" in text
    assert "mobile on Verizon Wireless" in text
    assert "Akron, OH" in text
    assert "never treat it as confirmed" in text
    assert lookup_text(NumberInfo()) == ""
