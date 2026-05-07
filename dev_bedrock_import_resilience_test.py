from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

from pydantic import BaseModel

SOURCE_PATH = Path(__file__).with_name("bedrock_llms.py")


class _FakePlugins:
    def __init__(self):
        self.calls: list[str] = []
        self._plugin = types.SimpleNamespace(
            load_settings=lambda: {"Anthropic Claude 3 Sonnet": True}
        )

    def get(self, name: str):
        self.calls.append(name)
        if name == "amazon-bedrock-llms":
            return self._plugin
        return None


class _FakeMadHatter:
    def __init__(self, plugins: _FakePlugins):
        self.plugins = plugins


class _FakeBoto3:
    calls: list[str] = []

    def get_client(self, service_name: str):
        self.calls.append(service_name)
        if service_name == "bedrock":
            return _FailingBedrockClient()
        return types.SimpleNamespace(meta=types.SimpleNamespace(region_name="eu-west-1"))


class _FailingBedrockClient:
    def list_foundation_models(self, **kwargs):
        raise RuntimeError("bedrock discovery unavailable during bootstrap")

    def list_guardrails(self, **kwargs):
        raise RuntimeError("guardrails unavailable during bootstrap")


class _FakeBaseMessage:
    def __init__(self, content=None, **kwargs):
        self.content = content
        self.usage_metadata = kwargs.get("usage_metadata", {})


class _FakeAIMessage(_FakeBaseMessage):
    pass


class _FakeHumanMessage(_FakeBaseMessage):
    pass


class _FakeSystemMessage(_FakeBaseMessage):
    pass


class _FakeChatPromptValue:
    def __init__(self, messages):
        self.messages = messages


class _FakeLangchainModel:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def invoke(self, *args, **kwargs):
        return types.SimpleNamespace(content="ok", usage_metadata={})


class _FakeLogger:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None


class _FakeCrudModule:
    @staticmethod
    def get_setting_by_name(name: str):
        return {"value": {"name": "Anthropic Claude 3 Sonnet", "model_id": "arn:test/anthropic.claude-3-sonnet-20240229-v1:0"}}


class _FakeLLMSettings(BaseModel):
    pass


def _identity_decorator(*args, **kwargs):
    if args and callable(args[0]) and len(args) == 1 and not kwargs:
        return args[0]

    def decorator(func):
        return func

    return decorator


def _install_module(name: str, **attributes):
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    sys.modules[name] = module
    return module


def _install_stubs() -> _FakePlugins:
    fake_plugins = _FakePlugins()

    _install_module("cat")
    _install_module("cat.db", crud=_FakeCrudModule)
    _install_module("cat.log", log=_FakeLogger())
    _install_module(
        "cat.mad_hatter.decorators",
        tool=_identity_decorator,
        hook=_identity_decorator,
        plugin=_identity_decorator,
    )
    _install_module(
        "cat.mad_hatter.mad_hatter",
        MadHatter=lambda: _FakeMadHatter(fake_plugins),
    )
    _install_module("cat.factory.llm", LLMSettings=_FakeLLMSettings)
    _install_module("cat.plugins.aws_integration", Boto3=_FakeBoto3)

    _install_module("langchain_aws", BedrockLLM=_FakeLangchainModel, ChatBedrock=_FakeLangchainModel)
    _install_module(
        "langchain_core.messages",
        AIMessage=_FakeAIMessage,
        BaseMessage=_FakeBaseMessage,
        HumanMessage=_FakeHumanMessage,
        SystemMessage=_FakeSystemMessage,
    )
    _install_module("langchain_core.prompt_values", ChatPromptValue=_FakeChatPromptValue)

    def parse_pricing_with_model(model_names, model_id, client):
        mapping = {
            "anthropic.claude-3-sonnet": "Anthropic Claude 3 Sonnet",
            "anthropic.claude-3-haiku": "Anthropic Claude 3 Haiku",
            "amazon.titan-text-lite": "Amazon Titan Text G1 - Lite",
            "amazon.titan-text-express": "Amazon Titan Text G1 - Express",
            "mistral.mistral-7b-instruct": "Mistral AI Mistral 7B Instruct",
            "mistral.mixtral-8x7b-instruct": "Mistral AI Mixtral 8x7B Instruct",
            "mistral.mistral-large-2402": "Mistral AI Mistral Large (24,02)",
        }
        for prefix, name in mapping.items():
            if model_id.startswith(prefix) and name in model_names:
                return name
        return "Error"

    _install_module(
        "bedrock_price_estimator",
        fetch_aws_pricing=lambda client: [],
        parse_pricing_with_model=parse_pricing_with_model,
        get_model_names=lambda pricing_data: [],
        filter_pricing_by_model=lambda pricing_data, model_name: [],
        extract_model_pricing=lambda filtered_pricing, model_id: {model_id: {"input": {}, "output": {}, "cache_read_input": {}}},
    )

    return fake_plugins


def _load_module():
    spec = importlib.util.spec_from_file_location("bedrock_llms_import_smoke", SOURCE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> int:
    fake_plugins = _install_stubs()
    module = _load_module()

    assert _FakeBoto3.calls == [], "l'import del plugin non deve inizializzare client AWS"
    assert fake_plugins.calls == [], "l'import del plugin non deve dipendere dal registry dei plugin"

    loaded_settings = module._load_plugin_settings()
    assert loaded_settings == {"Anthropic Claude 3 Sonnet": True}
    assert "amazon-bedrock-llms" in fake_plugins.calls, "deve cercare anche il nome plugin con trattini"
    assert _FakeBoto3.calls == [], "caricare le settings del plugin non deve toccare AWS"

    cached_models = module.get_cached_available_models()
    assert "Anthropic Claude 3 Sonnet" in cached_models, "il catalogo cached deve preservare i modelli Bedrock"

    SettingsModel = module.get_settings()
    defaults = SettingsModel().model_dump()
    assert defaults, "la settings model non deve essere vuota quando la discovery live fallisce"
    assert "Anthropic Claude 3 Sonnet" in defaults
    assert any(defaults.values()), "almeno un modello Bedrock deve restare selezionabile di default"
    assert _FakeBoto3.calls.count("bedrock") == 1, "il client bedrock deve essere lazy e cache-ato"

    allowed_llms = module.factory_pipeline()
    assert allowed_llms, "factory_pipeline deve restituire almeno un LLM dalla cache quando bootstrap live fallisce"
    assert any(
        llm.model_config["json_schema_extra"]["humanReadableName"] == "Amazon Bedrock: Anthropic Claude 3 Sonnet"
        for llm in allowed_llms
    ), "le settings salvate del plugin devono ancora selezionare il modello Bedrock desiderato"

    print("Bedrock import resilience: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


