#!/usr/bin/env python3
"""Patch glm4_moe_detector.py: make the mandatory newline after the function
name optional, so the GLM-5.3 no-newline variant of the same tag family parses
too. All tag tokens are built via chr() to avoid literal tags in this file.

v0.5.21 port note: identical to the v0520_fix version except the unit-test
sample s53 was rebuilt with the correct tag order (open arg_key before the
key text, close arg_key after it); the v0520 copy had them swapped, which
made its self-test fail even though the patch itself was correct (verified
end-to-end by verify_openclaw 12/12 in stage 4)."""
from pathlib import Path
import re

LT, GT = chr(60), chr(62)
BOT = LT + "tool_call" + GT
EOT = LT + "/tool_call" + GT
AKO = LT + "arg_key" + GT
NL = r"(?:\\n|\n)"
NEW = NL[:-1] + "|(?=" + AKO + ")|(?=" + EOT + "))"

p = Path(__file__).parent / "python/sglang/srt/function_call/glm4_moe_detector.py"
s = p.read_text(encoding="utf-8")

bot = re.search(r'self\.bot_token = "([^"]+)"', s).group(1)
eot = re.search(r'self\.eot_token = "([^"]+)"', s).group(1)
assert bot == BOT and eot == EOT, f"unexpected tokens: {bot!r} {eot!r}"

# Revert a previous half-applied patch if present
if NEW in s:
    s = s.replace(NEW, NL)
    print("reverted previous patch")

m1 = re.search(r'(_STREAMING_PARTIAL_PATTERN = re\.compile\(\s*r")([^"]+)(")', s)
assert m1 and NL in m1.group(2), "stream pattern not found"
s = s[: m1.start(2)] + m1.group(2).replace(NL, NEW) + s[m1.end(2) :]

m2 = re.search(r'(self\.func_detail_regex = re\.compile\(\s*r")([^"]+)(")', s)
assert m2 and NL in m2.group(2), "detail pattern not found"
s = s[: m2.start(2)] + m2.group(2).replace(NL, NEW) + s[m2.end(2) :]

p.write_text(s, encoding="utf-8")
print("patched OK")

# ---- unit tests ----
r1 = re.compile(m1.group(2).replace(NL, NEW), re.DOTALL)
r2 = re.compile(m2.group(2).replace(NL, NEW), re.DOTALL)
arg_src = re.search(r'self\.func_arg_regex = re\.compile\(\s*r"([^"]+)"', s).group(1)
arg_re = re.compile(arg_src, re.DOTALL)
AKC = LT + "/arg_key" + GT
AVO = LT + "arg_value" + GT
AVC = LT + "/arg_value" + GT

# GLM-5.3 (no newline): open-arg_key tag BEFORE key text, close tag AFTER it.
s53 = BOT + "get_weather" + AKO + "city" + AKC + AVO + "Beijing" + AVC + EOT
g = r2.search(s53)
assert g and g.group(1) == "get_weather" and "city" in g.group(2), f"glm53 fail: {g}"
print("glm53 detail OK:", g.group(1), "|", g.group(2)[:60])

s45 = (
    BOT + "get_weather" + chr(10) + AKC + "city" + AKO + chr(10)
    + AVO + "Beijing" + AVC + chr(10) + EOT
)
g2 = r2.search(s45)
assert g2 and g2.group(1) == "get_weather", "glm45 regression"
print("glm45 backward-compat OK")

p1 = r1.search(s53)
assert p1 and p1.group(1) == "get_weather", f"stream fail: {p1 and p1.groups()}"
print("stream partial OK:", p1.groups()[:2])

am = arg_re.findall(g.group(2))
assert am and am[0] == ("city", "Beijing"), f"arg fail: {am}"
print("arg extract OK:", am)

g3 = r2.search(BOT + "list_all" + EOT)
assert g3 and g3.group(1) == "list_all", "no-arg fail"
print("no-arg OK")
print("=== ALL REGEX UNIT TESTS PASSED ===")
