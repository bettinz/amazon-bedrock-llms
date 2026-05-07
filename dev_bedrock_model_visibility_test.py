from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path
from typing import Any

SOURCE_PATH = Path(__file__).with_name("bedrock_llms.py")
TARGET_FUNCTIONS = {"_get_model_identifier", "get_availale_models"}


class _FakeLogger:
    def warning(self, *args, **kwargs):
        return None


class _FakeClient:
    def list_foundation_models(self, **kwargs):
        return {
            "modelSummaries": [
                {
                    "modelArn": "arn:aws:bedrock:eu-west-1::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0",
                    "modelId": "anthropic.claude-3-sonnet-20240229-v1:0",
                    "providerName": "Anthropic",
                    "modelName": "Claude 3 Sonnet",
                    "responseStreamingSupported": True,
                }
            ]
        }


def _load_namespace():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(SOURCE_PATH))
    selected_nodes = [
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name in TARGET_FUNCTIONS
    ]
    compiled = compile(ast.Module(body=selected_nodes, type_ignores=[]), str(SOURCE_PATH), "exec")
    namespace = {
        "Any": Any,
        "defaultdict": defaultdict,
        "logger": _FakeLogger(),
    }
    exec(compiled, namespace)
    return namespace


def main() -> int:
    namespace = _load_namespace()
    get_availale_models = namespace["get_availale_models"]

    namespace["_get_pricing_catalog"] = lambda: ([{"model": "Claude 3 Sonnet"}], ["Claude 3 Sonnet"])
    namespace["get_or_update_pricing"] = lambda *args, **kwargs: {"error": "pricing unavailable"}
    namespace["save_pricing_cache"] = lambda: None

    models = get_availale_models(_FakeClient())
    assert "Anthropic Claude 3 Sonnet" in models, "il modello non deve sparire dalla discovery se il pricing fallisce"
    payload = models["Anthropic Claude 3 Sonnet"][0]
    assert payload["pricing_info"] == {
        "anthropic.claude-3-sonnet-20240229-v1:0": {"input": {}, "output": {}, "cache_read_input": {}}
    }

    print("Bedrock model visibility fallback: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

