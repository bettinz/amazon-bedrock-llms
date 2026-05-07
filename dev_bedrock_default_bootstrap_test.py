from __future__ import annotations

import ast
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, create_model

SOURCE_PATH = Path(__file__).with_name("bedrock_llms.py")
TARGET_FUNCTION = "create_dynamic_model"


def _load_create_dynamic_model():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(SOURCE_PATH))
    selected_nodes = [
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == TARGET_FUNCTION
    ]
    compiled = compile(ast.Module(body=selected_nodes, type_ignores=[]), str(SOURCE_PATH), "exec")
    namespace = {
        "__name__": "bedrock_default_bootstrap_test",
        "BaseModel": BaseModel,
        "ConfigDict": ConfigDict,
        "Field": Field,
        "create_model": create_model,
        "DEFAULT_MODEL": "amazon.titan-tg1-large",
    }
    exec(compiled, namespace)
    return namespace[TARGET_FUNCTION]


def main() -> int:
    create_dynamic_model = _load_create_dynamic_model()

    amazon_llms = {
        "Anthropic Claude 3 Sonnet": [{"model_arn": "arn:aws:bedrock:eu-west-1::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0"}],
        "Anthropic Claude 3 Haiku": [{"model_arn": "arn:aws:bedrock:eu-west-1::foundation-model/anthropic.claude-3-haiku-20240307-v1:0"}],
    }

    DynamicModel = create_dynamic_model(amazon_llms)
    defaults = DynamicModel().model_dump()

    assert defaults["Anthropic Claude 3 Sonnet"] is True, "il primo modello deve diventare default se DEFAULT_MODEL non corrisponde a nulla"
    assert defaults["Anthropic Claude 3 Haiku"] is False

    effective_settings = {**defaults, **{}}
    hydrated = DynamicModel(**effective_settings).model_dump()
    assert hydrated["Anthropic Claude 3 Sonnet"] is True, "con settings vuote il default non deve andare perso"

    print("Bedrock default bootstrap: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


