from __future__ import annotations

import ast
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

SOURCE_PATH = Path(__file__).with_name("bedrock_llms.py")
TARGET_FUNCTIONS = {
    "_utcnow",
    "_is_cache_fresh",
    "_get_model_identifier",
    "_get_pricing_catalog",
    "get_or_update_pricing",
}


class _FakeLogger:
    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None


class _FetchCounter:
    def __init__(self):
        self.calls = 0

    def __call__(self, client: Any):
        self.calls += 1
        return [{"model": "Claude 3 Sonnet"}, {"model": "Nova Pro"}]


class _ParseCounter:
    def __init__(self):
        self.calls = 0

    def __call__(self, model_names: list[str], model_id: str, client: Any):
        self.calls += 1
        return "Claude 3 Sonnet"


class _FakeLock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _load_functions():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(SOURCE_PATH))
    selected_nodes = [
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name in TARGET_FUNCTIONS
    ]
    compiled = compile(ast.Module(body=selected_nodes, type_ignores=[]), str(SOURCE_PATH), "exec")
    namespace = {
        "Any": Any,
        "datetime": datetime,
        "timedelta": timedelta,
        "timezone": timezone,
        "pricing_cache": {},
        "_pricing_catalog_cache": {"data": None, "model_names": None, "timestamp": None},
        "_pricing_cache_dirty": False,
        "pricing_client": object(),
        "bedrock_runtime_client": object(),
        "logger": _FakeLogger(),
        "Path": Path,
        "json": json,
        "os": __import__("os"),
        "tempfile": __import__("tempfile"),
        "Lock": _FakeLock,
    }
    exec(compiled, namespace)
    return namespace


def main() -> int:
    namespace = _load_functions()
    get_model_identifier = namespace["_get_model_identifier"]
    get_pricing_catalog = namespace["_get_pricing_catalog"]
    get_or_update_pricing = namespace["get_or_update_pricing"]

    fetch_counter = _FetchCounter()
    parse_counter = _ParseCounter()
    namespace["fetch_aws_pricing"] = fetch_counter
    namespace["get_model_names"] = lambda pricing_data: [item["model"] for item in pricing_data]
    namespace["parse_pricing_with_model"] = parse_counter
    namespace["filter_pricing_by_model"] = lambda pricing_data, model_name: [item for item in pricing_data if item["model"] == model_name]
    namespace["extract_model_pricing"] = lambda filtered_pricing, model_id: {model_id: {"input": {"price": 1}}}

    model_identifier = get_model_identifier({"modelArn": "arn:aws:bedrock:::model/test", "modelId": "test-model"})
    assert model_identifier == "arn:aws:bedrock:::model/test", f"identifier inatteso: {model_identifier}"
    assert get_model_identifier({"modelId": "fallback-model"}) == "fallback-model"

    pricing_data, model_names = get_pricing_catalog()
    assert fetch_counter.calls == 1, f"attesa una fetch pricing, ottenute {fetch_counter.calls}"
    assert model_names == ["Claude 3 Sonnet", "Nova Pro"]

    pricing_data_2, model_names_2 = get_pricing_catalog()
    assert fetch_counter.calls == 1, "la seconda lettura della pricing catalog cache non deve rifare il fetch"
    assert pricing_data_2 == pricing_data
    assert model_names_2 == model_names

    first = get_or_update_pricing("anthropic.claude-sonnet", "arn:model:1", pricing_data, model_names)
    second = get_or_update_pricing("anthropic.claude-sonnet", "arn:model:1", pricing_data, model_names)
    assert parse_counter.calls == 1, "la seconda pricing lookup deve usare la cache per model key"
    assert first == second

    print("Bedrock cache optimizations: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


