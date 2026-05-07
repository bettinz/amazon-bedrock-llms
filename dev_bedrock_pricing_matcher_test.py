from __future__ import annotations

import ast
import difflib
import re
from pathlib import Path
from typing import Any, List

SOURCE_PATH = Path(__file__).with_name("bedrock_price_estimator.py")
TARGET_FUNCTIONS = {
    "_normalize_identifier",
    "_tokenize_identifier",
    "_extract_provider",
    "_score_model_name_candidate",
    "_resolve_model_name_locally",
    "parse_pricing_with_model",
}


class _FakeLogger:
    def info(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None


class _FakeClientError(Exception):
    pass


def _load_matcher_namespace():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(SOURCE_PATH))
    selected_nodes = [
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name in TARGET_FUNCTIONS
    ]
    compiled = compile(ast.Module(body=selected_nodes, type_ignores=[]), str(SOURCE_PATH), "exec")
    namespace = {
        "Any": Any,
        "List": List,
        "re": re,
        "difflib": difflib,
        "logger": _FakeLogger(),
        "botocore": type("_Botocore", (), {"exceptions": type("_Exc", (), {"ClientError": _FakeClientError})})(),
        "_GENERIC_TOKENS": {
            "amazon", "anthropic", "mistral", "meta", "cohere", "ai21", "stability", "ai",
            "model", "foundation", "inference", "profile", "global", "us", "eu", "apac", "v", "text",
        },
        "_FAMILY_TOKENS": {"claude", "nova", "titan", "mistral", "mixtral", "llama", "command", "jurassic", "jamba", "embed"},
    }
    exec(compiled, namespace)
    return namespace


def main() -> int:
    ns = _load_matcher_namespace()
    parse_pricing_with_model = ns["parse_pricing_with_model"]
    normalize_identifier = ns["_normalize_identifier"]

    model_names = [
        "Anthropic Claude 3 Sonnet",
        "Anthropic Claude 3 Haiku",
        "Amazon Titan Text G1 - Lite",
        "Amazon Titan Text G1 - Express",
        "Mistral AI Mistral 7B Instruct",
        "Mistral AI Mixtral 8x7B Instruct",
        "Mistral AI Mistral Large (24,02)",
        "Amazon Nova Pro",
    ]

    assert normalize_identifier("arn:aws:bedrock:eu-west-1::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0") == "anthropic claude 3 sonnet 20240229"

    assert parse_pricing_with_model(model_names, "anthropic.claude-3-sonnet-20240229-v1:0", object()) == "Anthropic Claude 3 Sonnet"
    assert parse_pricing_with_model(model_names, "anthropic.claude-3-haiku-20240307-v1:0", object()) == "Anthropic Claude 3 Haiku"
    assert parse_pricing_with_model(model_names, "amazon.titan-text-lite-v1", object()) == "Amazon Titan Text G1 - Lite"
    assert parse_pricing_with_model(model_names, "amazon.titan-text-express-v1", object()) == "Amazon Titan Text G1 - Express"
    assert parse_pricing_with_model(model_names, "mistral.mistral-7b-instruct-v0:2", object()) == "Mistral AI Mistral 7B Instruct"
    assert parse_pricing_with_model(model_names, "mistral.mixtral-8x7b-instruct-v0:1", object()) == "Mistral AI Mixtral 8x7B Instruct"
    assert parse_pricing_with_model(model_names, "mistral.mistral-large-2402-v1:0", object()) == "Mistral AI Mistral Large (24,02)"
    assert parse_pricing_with_model(model_names, "amazon.nova-pro-v1:0", object()) == "Amazon Nova Pro"

    print("Bedrock pricing matcher: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

