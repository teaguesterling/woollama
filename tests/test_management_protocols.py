"""Tests for the `management_protocol` selector field (Inferencer) and the
`[management_protocols.<name>]` config parser (config.load_management_protocols).

Pure data-layer: no backend construction, no lifespan wiring — this mirrors the
already-merged Rust `EndpointSpec`/`ProtocolSpec` + `load_management_protocols`
in woollama-engine/src/lib.rs. Task 3 consumes what's produced here.
"""
from __future__ import annotations

import pytest

from woollama import config, inferencers


@pytest.fixture(autouse=True)
def _clean_config_dir(monkeypatch, tmp_path):
    """Point config at an empty dir so a real user inferencers.toml can't leak
    into these tests; a test that wants config writes tmp_path/inferencers.toml."""
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))


TOML = '''
[inferencers.dev]
base_url = "http://box:8000/v1"
management_url = "http://box:8800"
management_protocol = "mybox"

[management_protocols.mybox]
kind = "rest"

[management_protocols.mybox.endpoints.running]
url = "http://box:8800/api/loaded"
path = "models"
id_field = "id"

[management_protocols.mybox.endpoints.start]
url = "http://box:8800/api/load"
method = "POST"
body = '{"model": "{id}"}'

[management_protocols.mybox.endpoints.stop]
url = "http://box:8800/api/unload"
method = "POST"
headers = { Authorization = "Bearer ${MYBOX_TOKEN}" }

[management_protocols.oll]
kind = "ollama"
keep_alive = "5m"
'''


def test_inferencer_management_protocol_field_default():
    # Back-compat: absent -> None, the "use the built-in tiiny backend" sentinel.
    inf = inferencers.Inferencer(name="x", base_url="http://x/v1")
    assert inf.management_protocol is None


def test_inferencer_picks_up_management_protocol(tmp_path, monkeypatch):
    monkeypatch.setenv("MYBOX_TOKEN", "secret-tok")
    (tmp_path / "inferencers.toml").write_text(TOML)
    dev = inferencers.get("dev")
    assert dev.management_protocol == "mybox"
    assert dev.management_url == "http://box:8800"


def test_load_management_protocols_absent_is_empty():
    assert config.load_management_protocols() == {}


def test_load_management_protocols_rest_and_ollama(tmp_path, monkeypatch):
    monkeypatch.setenv("MYBOX_TOKEN", "secret-tok")
    (tmp_path / "inferencers.toml").write_text(TOML)
    protos = config.load_management_protocols()
    assert set(protos) == {"mybox", "oll"}

    mybox = protos["mybox"]
    assert mybox.kind == "rest"
    assert mybox.running.url == "http://box:8800/api/loaded"
    assert mybox.running.path == "models"
    assert mybox.running.id_field == "id"
    assert mybox.running.method is None            # no default templating/method here
    assert mybox.start.url == "http://box:8800/api/load"
    assert mybox.start.method == "POST"
    assert mybox.start.body == '{"model": "{id}"}'
    assert mybox.start.path is None                # path only applies to `running`
    assert mybox.stop.url == "http://box:8800/api/unload"
    # ${VAR} IS expanded (same _expand_env pass as the rest of inferencers.toml).
    assert mybox.stop.headers == {"Authorization": "Bearer secret-tok"}

    oll = protos["oll"]
    assert oll.kind == "ollama"
    assert oll.keep_alive == "5m"


def test_load_management_protocols_unknown_kind(tmp_path):
    (tmp_path / "inferencers.toml").write_text(
        '[management_protocols.bad]\nkind = "carrier-pigeon"\n')
    with pytest.raises(ValueError, match="bad"):
        config.load_management_protocols()


def test_load_management_protocols_rest_missing_stop(tmp_path):
    (tmp_path / "inferencers.toml").write_text(
        '[management_protocols.mybox]\n'
        'kind = "rest"\n\n'
        '[management_protocols.mybox.endpoints.running]\n'
        'url = "http://box:8800/api/loaded"\n'
        'path = "models"\n\n'
        '[management_protocols.mybox.endpoints.start]\n'
        'url = "http://box:8800/api/load"\n')
    with pytest.raises(ValueError, match="mybox"):
        config.load_management_protocols()


def test_load_management_protocols_optional_fields_nonstring_coerce_to_none(tmp_path):
    """Parity pin: the Rust authority (woollama-engine/src/lib.rs) parses
    method/body/id_field/keep_alive with `.and_then(Value::as_str)`, which
    SILENTLY treats a present-but-non-string value as absent (None) rather than
    erroring. Only required fields (url, path) reject wrong types. This must
    not raise."""
    (tmp_path / "inferencers.toml").write_text(
        '[management_protocols.weird]\n'
        'kind = "rest"\n\n'
        '[management_protocols.weird.endpoints.running]\n'
        'url = "http://x/loaded"\n'
        'path = "models"\n'
        'id_field = 123\n\n'                        # wrong type, optional
        '[management_protocols.weird.endpoints.start]\n'
        'url = "http://x/load"\n'
        'method = 42\n'                              # wrong type, optional
        'body = true\n\n'                             # wrong type, optional
        '[management_protocols.weird.endpoints.stop]\n'
        'url = "http://x/unload"\n\n'
        '[management_protocols.weird2]\n'
        'kind = "ollama"\n'
        'keep_alive = 99\n')                          # wrong type, optional

    protos = config.load_management_protocols()
    assert protos["weird"].running.id_field is None
    assert protos["weird"].start.method is None
    assert protos["weird"].start.body is None
    assert protos["weird2"].keep_alive is None


def test_load_management_protocols_running_requires_path(tmp_path):
    (tmp_path / "inferencers.toml").write_text(
        '[management_protocols.mybox]\n'
        'kind = "rest"\n\n'
        '[management_protocols.mybox.endpoints.running]\n'
        'url = "http://box:8800/api/loaded"\n\n'
        '[management_protocols.mybox.endpoints.start]\n'
        'url = "http://box:8800/api/load"\n\n'
        '[management_protocols.mybox.endpoints.stop]\n'
        'url = "http://box:8800/api/unload"\n')
    with pytest.raises(ValueError, match="path"):
        config.load_management_protocols()
