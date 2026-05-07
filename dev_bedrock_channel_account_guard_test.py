from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any, Type, cast

from pydantic import BaseModel

SOURCE_PATH = Path(__file__).with_name("bedrock_llms.py")


class _FakePlugins:
    def __init__(self):
        self._plugin = types.SimpleNamespace(
            load_settings=lambda: {"Anthropic Claude 3 Sonnet": True}
        )

    def get(self, name: str):
        if name == "amazon-bedrock-llms":
            return self._plugin
        return None


class _FakeMadHatter:
    def __init__(self, plugins: _FakePlugins):
        self.plugins = plugins


class _FakeBoto3Session:
    def client(self, service_name: str, **kwargs):
        if service_name == "bedrock":
            return _FailingBedrockClient()
        return types.SimpleNamespace(meta=types.SimpleNamespace(region_name="eu-west-1"))


class _FakeBoto3Module:
    @staticmethod
    def Session(**kwargs):
        return _FakeBoto3Session()


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
        return {
            "value": {
                "name": "Anthropic Claude 3 Sonnet",
                "model_id": "arn:test/anthropic.claude-3-sonnet-20240229-v1:0",
            }
        }


class _FakeLLMSettings(BaseModel):
    _pyclass: Type = None

    @classmethod
    def get_llm_from_config(cls, config):
        if cls._pyclass is None:
            raise Exception("_pyclass must not be None")
        return cast(Any, cls._pyclass).default(**config)


class _RestrictedChatBedrock:
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs

    def invoke(self, *args, **kwargs):
        raise ValueError(
            "Error raised by bedrock service: An error occurred (ValidationException) when calling the InvokeModelWithResponseStream operation: "
            "Access to this model is not available for channel program accounts. Reach out to your AWS Solution Provider or AWS Distributor for more information."
        )


class _FakeBedrockLLM(_RestrictedChatBedrock):
    pass


class _FakeChatBedrockConverse(_RestrictedChatBedrock):
    pass


class _FakeCoreLLMConfig(BaseModel):
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


def _install_stubs():
    fake_plugins = _FakePlugins()
    _install_module("cat")
    _install_module("cat.db", crud=_FakeCrudModule)
    _install_module("cat.log", log=_FakeLogger())
    _install_module("cat.factory")
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
    _install_module(
        "cat.factory.llm",
        LLMSettings=_FakeLLMSettings,
        get_allowed_language_models=lambda: [_FakeCoreLLMConfig],
    )
    _install_module("boto3", Session=_FakeBoto3Module.Session)
    _install_module(
        "langchain_core.messages",
        AIMessage=_FakeAIMessage,
        BaseMessage=_FakeBaseMessage,
        HumanMessage=_FakeHumanMessage,
        SystemMessage=_FakeSystemMessage,
    )
    _install_module("langchain_core.prompt_values", ChatPromptValue=_FakeChatPromptValue)
    _install_module(
        "langchain_aws",
        BedrockLLM=_FakeBedrockLLM,
        ChatBedrock=_RestrictedChatBedrock,
        ChatBedrockConverse=_FakeChatBedrockConverse,
    )

    def parse_pricing_with_model(model_names, model_id, client):
        if model_id.startswith("anthropic.claude-3-sonnet"):
            return "Anthropic Claude 3 Sonnet"
        return "Error"

    _install_module(
        "bedrock_price_estimator",
        fetch_aws_pricing=lambda client: [],
        parse_pricing_with_model=parse_pricing_with_model,
        get_model_names=lambda pricing_data: [],
        filter_pricing_by_model=lambda pricing_data, model_name: [],
        extract_model_pricing=lambda filtered_pricing, model_id: {
            model_id: {"input": {}, "output": {}, "cache_read_input": {}}
        },
    )


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "bedrock_channel_guard_smoke", SOURCE_PATH
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> int:
    _install_stubs()
    module = _load_module()

    allowed_llms = module.factory_pipeline()
    selected_llm = next(
        llm
        for llm in allowed_llms
        if llm.model_config["json_schema_extra"]["humanReadableName"]
        == "Amazon Bedrock: Anthropic Claude 3 Sonnet"
    )

    runtime_llm = selected_llm.get_llm_from_config(selected_llm().model_dump())
    response = runtime_llm.invoke("hello")

    assert isinstance(response, _FakeAIMessage)
    assert "not available for your AWS account" in response.content
    assert "channel-program entitlement restriction" in response.content

    print("Bedrock channel account guard: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



