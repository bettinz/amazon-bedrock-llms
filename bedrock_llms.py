import os
import re
import json
import tempfile
from pathlib import Path
from threading import Lock
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from enum import Enum
from typing import Any, ClassVar, List, Optional, Type, cast
from cat.db import crud

from pydantic import BaseModel, model_validator, Field, create_model, ConfigDict
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompt_values import ChatPromptValue
import logging

from cat.log import log
from cat.mad_hatter.decorators import tool, hook, plugin
from cat.mad_hatter.mad_hatter import MadHatter
from cat.factory.llm import LLMSettings
import cat.factory.llm as cat_llm_factory

try:
    from cat.plugins.aws_integration import Boto3 as _CatPluginBoto3
except ImportError:
    _CatPluginBoto3 = None

if _CatPluginBoto3 is None:
    import boto3

    class Boto3:
        _session = None
        _clients: dict[tuple[str, str | None], Any] = {}

        @classmethod
        def _get_region_name(cls, service_name: str) -> str | None:
            if service_name == "pricing":
                return (
                    os.getenv("AWS_PRICING_REGION")
                    or os.getenv("AWS_BEDROCK_REGION")
                    or os.getenv("AWS_REGION")
                    or os.getenv("AWS_DEFAULT_REGION")
                    or "us-east-1"
                )

            return (
                os.getenv("AWS_BEDROCK_REGION")
                or os.getenv("AWS_REGION")
                or os.getenv("AWS_DEFAULT_REGION")
            )

        @classmethod
        def _get_session(cls):
            if cls._session is None:
                session_kwargs = {}
                profile_name = os.getenv("AWS_PROFILE")
                default_region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")

                if profile_name:
                    session_kwargs["profile_name"] = profile_name
                if default_region:
                    session_kwargs["region_name"] = default_region

                cls._session = boto3.Session(**session_kwargs)

            return cls._session

        def get_client(self, service_name: str):
            region_name = self._get_region_name(service_name)
            cache_key = (service_name, region_name)

            if cache_key not in self._clients:
                client_kwargs = {}
                if region_name:
                    client_kwargs["region_name"] = region_name
                self._clients[cache_key] = self._get_session().client(
                    service_name, **client_kwargs
                )

            return self._clients[cache_key]
else:
    Boto3 = _CatPluginBoto3

try:
    from .bedrock_price_estimator import (
        fetch_aws_pricing,
        parse_pricing_with_model,
        get_model_names,
        filter_pricing_by_model,
        extract_model_pricing,
    )
except ImportError:
    from bedrock_price_estimator import (
        fetch_aws_pricing,
        parse_pricing_with_model,
        get_model_names,
        filter_pricing_by_model,
        extract_model_pricing,
    )

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.info("Amazon Bedrock LLM plugin module imported.")

PLUGIN_NAME = "amazon_bedrock_llms"
DEFAULT_MODEL = "amazon.titan-tg1-large"
PLUGIN_ROOT = Path(__file__).resolve().parent

CACHED_PRICING_FILE = str(PLUGIN_ROOT / "cached_model_pricing.json")
CACHED_COST_FILE = str(PLUGIN_ROOT / "cached_model_costs.json")

client = None
pricing_client = None
bedrock_runtime_client = None

pricing_cache: dict[str, dict[str, Any]] = {}
_pricing_catalog_cache: dict[str, Any] = {"data": [], "model_names": [], "timestamp": None}
_pricing_cache_dirty = False
_current_model_cost_cache = None
_cost_cache_lock = Lock()

SEEDED_MODEL_CATALOG: dict[str, dict[str, str]] = {
    "Anthropic Claude 3 Sonnet": {
        "model_arn": "arn:aws:bedrock:eu-west-1::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0",
        "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
        "provider_name": "anthropic",
    },
    "Amazon Titan Text G1 - Lite": {
        "model_arn": "arn:aws:bedrock:eu-west-1::foundation-model/amazon.titan-text-lite-v1",
        "model_id": "amazon.titan-text-lite-v1",
        "provider_name": "amazon",
    },
    "Amazon Titan Text G1 - Express": {
        "model_arn": "arn:aws:bedrock:eu-west-1::foundation-model/amazon.titan-text-express-v1",
        "model_id": "amazon.titan-text-express-v1",
        "provider_name": "amazon",
    },
    "Anthropic Claude 3 Haiku": {
        "model_arn": "arn:aws:bedrock:eu-west-1::foundation-model/anthropic.claude-3-haiku-20240307-v1:0",
        "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
        "provider_name": "anthropic",
    },
    "Mistral AI Mistral 7B Instruct": {
        "model_arn": "arn:aws:bedrock:eu-west-1::foundation-model/mistral.mistral-7b-instruct-v0:2",
        "model_id": "mistral.mistral-7b-instruct-v0:2",
        "provider_name": "mistral",
    },
    "Mistral AI Mixtral 8x7B Instruct": {
        "model_arn": "arn:aws:bedrock:eu-west-1::foundation-model/mistral.mixtral-8x7b-instruct-v0:1",
        "model_id": "mistral.mixtral-8x7b-instruct-v0:1",
        "provider_name": "mistral",
    },
    "Mistral AI Mistral Large (24,02)": {
        "model_arn": "arn:aws:bedrock:eu-west-1::foundation-model/mistral.mistral-large-2402-v1:0",
        "model_id": "mistral.mistral-large-2402-v1:0",
        "provider_name": "mistral",
    },
}


def _get_bedrock_client():
    global client
    if client is None:
        client = Boto3().get_client("bedrock")
    return client


def _get_pricing_client():
    global pricing_client
    if pricing_client is None:
        pricing_client = Boto3().get_client("pricing")
    return pricing_client


def _get_bedrock_runtime_client():
    global bedrock_runtime_client
    if bedrock_runtime_client is None:
        bedrock_runtime_client = Boto3().get_client("bedrock-runtime")
    return bedrock_runtime_client


def _get_plugin_name_candidates() -> list[str]:
    folder_name = PLUGIN_ROOT.name
    return list(
        dict.fromkeys(
            [
                PLUGIN_NAME,
                folder_name,
                folder_name.replace("-", "_"),
                folder_name.replace("_", "-"),
            ]
        )
    )


def _get_plugin_instance():
    try:
        plugins = getattr(MadHatter(), "plugins", None)
        if plugins is None:
            return None

        for candidate in _get_plugin_name_candidates():
            plugin_instance = plugins.get(candidate)
            if plugin_instance is not None:
                return plugin_instance
    except Exception as e:
        logger.warning(f"Unable to resolve plugin instance from MadHatter: {e}")

    return None


def _load_plugin_settings() -> dict[str, Any]:
    plugin_instance = _get_plugin_instance()
    if plugin_instance is None:
        return {}

    try:
        return plugin_instance.load_settings() or {}
    except Exception as e:
        logger.warning(f"Unable to load plugin settings, using defaults: {e}")
        return {}


def _merge_llm_config_classes(base_llms: list[type], extra_llms: list[type]) -> list[type]:
    merged_llms = list(base_llms)
    existing_names = {llm.__name__ for llm in merged_llms}

    for llm in extra_llms:
        if llm.__name__ not in existing_names:
            merged_llms.append(llm)
            existing_names.add(llm.__name__)

    return merged_llms


def _get_bedrock_llm_configs_for_factory() -> list[type]:
    try:
        return factory_pipeline()
    except Exception as e:
        logger.warning(
            "Amazon Bedrock fallback LLM factory patch could not build model configs: %s",
            e,
        )
        return []


def _ensure_llm_factory_patch() -> None:
    original_get_allowed_language_models = getattr(
        cat_llm_factory, "_amazon_bedrock_original_get_allowed_language_models", None
    )

    if original_get_allowed_language_models is None:
        original_get_allowed_language_models = cat_llm_factory.get_allowed_language_models
        cat_llm_factory._amazon_bedrock_original_get_allowed_language_models = (
            original_get_allowed_language_models
        )

    if getattr(cat_llm_factory, "_amazon_bedrock_patch_installed", False):
        return

    def _patched_get_allowed_language_models():
        base_llms = original_get_allowed_language_models()
        bedrock_llms = _get_bedrock_llm_configs_for_factory()
        merged_llms = _merge_llm_config_classes(base_llms, bedrock_llms)
        logger.info(
            "Amazon Bedrock LLM factory patch returning %s total Cheshire Cat LLM configuration(s), including %s Bedrock configuration(s).",
            len(merged_llms),
            len(bedrock_llms),
        )
        return merged_llms

    cat_llm_factory.get_allowed_language_models = _patched_get_allowed_language_models
    cat_llm_factory._amazon_bedrock_patch_installed = True
    logger.info("Amazon Bedrock LLM factory patch installed.")


def _load_seed_model_names() -> list[str]:
    settings_path = PLUGIN_ROOT / "settings.json"
    if not settings_path.exists():
        return []

    try:
        with open(settings_path, "r", encoding="utf-8") as file:
            payload = json.load(file) or {}
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Unable to read seed model names from settings.json: {e}")
        return []

    return list(payload.keys())


def _humanize_model_id(model_id: str) -> str:
    tokens = [token for token in re.split(r"[._:-]+", model_id) if token and token.lower() not in {"v0", "v1", "v2"}]
    return " ".join(token.upper() if token.isdigit() or any(char.isdigit() for char in token) else token.title() for token in tokens)


def get_cached_available_models() -> dict[str, list[dict[str, Any]]]:
    if not os.path.exists(CACHED_PRICING_FILE):
        return {}

    try:
        with open(CACHED_PRICING_FILE, "r", encoding="utf-8") as file:
            cached_entries = json.load(file) or {}
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Unable to read cached Bedrock model catalog: {e}")
        return {}

    model_names = _load_seed_model_names()
    models = defaultdict(list)

    for model_arn, cached_entry in cached_entries.items():
        model_id = model_arn.split("/")[-1]
        provider_key = model_id.split(".")[0].lower()
        provider_name = "mistral" if "mistral" in provider_key else provider_key
        model_name = parse_pricing_with_model(model_names, model_id, object()) if model_names else "Error"
        if model_name == "Error":
            model_name = _humanize_model_id(model_id)

        pricing_info = cached_entry.get("data") or {model_id: {"input": {}, "output": {}, "cache_read_input": {}}}

        models[model_name].append(
            {
                "model_arn": model_arn,
                "provider_name": provider_name,
                "response_streaming_supported": True,
                "pricing_info": pricing_info,
                "model_id": model_id,
            }
        )

    return dict(models)


def get_seed_available_models() -> dict[str, list[dict[str, Any]]]:
    model_names = _load_seed_model_names()
    models = defaultdict(list)

    for model_name in model_names:
        seeded_model = SEEDED_MODEL_CATALOG.get(model_name)
        if seeded_model is None:
            continue

        model_id = seeded_model["model_id"]
        models[model_name].append(
            {
                "model_arn": seeded_model["model_arn"],
                "provider_name": seeded_model["provider_name"],
                "response_streaming_supported": True,
                "pricing_info": {model_id: {"input": {}, "output": {}, "cache_read_input": {}}},
                "model_id": model_id,
            }
        )

    return dict(models)


def _normalize_bedrock_messages(messages: list[BaseMessage]) -> tuple[list[BaseMessage], int]:
    normalized = list(messages)
    system_prefix: list[BaseMessage] = []

    while normalized and isinstance(normalized[0], SystemMessage):
        system_prefix.append(normalized.pop(0))

    dropped = 0
    while normalized and not isinstance(normalized[0], HumanMessage):
        if isinstance(normalized[0], AIMessage):
            normalized.pop(0)
            dropped += 1
            continue
        break

    if normalized and isinstance(normalized[0], HumanMessage):
        return [*system_prefix, *normalized], dropped

    return list(messages), 0


def _normalize_bedrock_input(model_input: Any) -> tuple[Any, int]:
    if isinstance(model_input, ChatPromptValue):
        normalized_messages, dropped = _normalize_bedrock_messages(list(model_input.messages))
        if dropped:
            return ChatPromptValue(messages=normalized_messages), dropped
        return model_input, 0

    if isinstance(model_input, (list, tuple)) and all(
        isinstance(message, BaseMessage) for message in model_input
    ):
        normalized_messages, dropped = _normalize_bedrock_messages(list(model_input))
        if dropped:
            return normalized_messages, dropped

    return model_input, 0


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _is_cache_fresh(timestamp: datetime | None, ttl: timedelta = timedelta(days=1)) -> bool:
    if timestamp is None:
        return False
    normalized_timestamp = timestamp
    if normalized_timestamp.tzinfo is None:
        normalized_timestamp = normalized_timestamp.replace(tzinfo=timezone.utc)
    return _utcnow() - normalized_timestamp < ttl


def _write_json_atomic(file_path: str, payload: dict) -> None:
    target = Path(file_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", delete=False, dir=target.parent, encoding="utf-8") as tmp_file:
        json.dump(payload, tmp_file, indent=4)
        tmp_name = tmp_file.name

    os.replace(tmp_name, target)


def _load_current_model_cost() -> float:
    global _current_model_cost_cache

    with _cost_cache_lock:
        if _current_model_cost_cache is not None:
            return _current_model_cost_cache

        if os.path.exists(CACHED_COST_FILE):
            try:
                with open(CACHED_COST_FILE, "r", encoding="utf-8") as file:
                    cost_data = json.load(file) or {}
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load model cost cache. Error: {e}")
                cost_data = {}
        else:
            cost_data = {}

        _current_model_cost_cache = float(cost_data.get("current_cost", 0.0))
        return _current_model_cost_cache


def _store_current_model_cost(cost_value: float) -> None:
    global _current_model_cost_cache

    with _cost_cache_lock:
        _current_model_cost_cache = float(cost_value)
        try:
            _write_json_atomic(CACHED_COST_FILE, {"current_cost": _current_model_cost_cache})
        except IOError as e:
            logger.error(f"Error saving model cost cache: {e}")


def _get_model_identifier(model_summary: dict[str, Any]) -> str | None:
    return model_summary.get("modelArn") or model_summary.get("modelId")


def _get_pricing_catalog(force_refresh: bool = False) -> tuple[list[Any], list[str]]:
    global _pricing_catalog_cache

    if (
        not force_refresh
        and _pricing_catalog_cache["data"] is not None
        and _is_cache_fresh(_pricing_catalog_cache["timestamp"])
    ):
        return cast(list[Any], _pricing_catalog_cache["data"]), cast(
            list[str], _pricing_catalog_cache["model_names"]
        )

    pricing_data = cast(
        list[Any], fetch_aws_pricing(pricing_client or _get_pricing_client())
    )
    model_names = get_model_names(cast(Any, pricing_data))
    _pricing_catalog_cache = {
        "data": pricing_data,
        "model_names": model_names,
        "timestamp": _utcnow(),
    }
    return cast(list[Any], pricing_data), model_names


def load_pricing_cache() -> None:
    """Loads the pricing cache from a JSON file if available."""
    global pricing_cache
    if os.path.exists(CACHED_PRICING_FILE):
        try:
            with open(CACHED_PRICING_FILE, "r", encoding="utf-8") as file:
                data = json.load(file)
                pricing_cache = {
                    model_id: {
                        "data": item["data"],
                        "timestamp": datetime.fromisoformat(item["timestamp"]),
                    }
                    for model_id, item in data.items()
                }
        except Exception as e:
            logger.warning(f"Failed to load pricing cache: {e}")
            pricing_cache = {}


def save_pricing_cache() -> None:
    """Saves the pricing cache to a JSON file."""
    global _pricing_cache_dirty

    if not _pricing_cache_dirty:
        return

    try:
        _write_json_atomic(
            CACHED_PRICING_FILE,
            {
                model_id: {
                    "data": item["data"],
                    "timestamp": item["timestamp"].isoformat(),
                }
                for model_id, item in pricing_cache.items()
            },
            )
        _pricing_cache_dirty = False
    except Exception as e:
        logger.warning(f"Failed to save pricing cache: {e}")


load_pricing_cache()


def get_or_update_pricing(
    model_id: str,
    model_cache_key: str,
    pricing_data: Optional[list[Any]] = None,
    model_names: Optional[list[str]] = None,
) -> dict[str, Any]:
    global _pricing_cache_dirty

    if model_cache_key in pricing_cache:
        cached_data = pricing_cache[model_cache_key]
        if _is_cache_fresh(cached_data["timestamp"]):
            return cached_data["data"]

    model_pricing = {model_id: {"input": {}, "output": {}, "cache_read_input": {}}}

    try:
        if pricing_data is None or model_names is None:
            pricing_data, model_names = _get_pricing_catalog()

        model_name = parse_pricing_with_model(
            model_names, model_id, bedrock_runtime_client or _get_bedrock_runtime_client()
        )

        if model_name == "Error":
            pricing_cache[model_cache_key] = {
                "data": model_pricing,
                "timestamp": _utcnow(),
            }
            _pricing_cache_dirty = True
            return {"error": "Failed to determine model name."}

        filtered_pricing = filter_pricing_by_model(pricing_data, model_name)

        if filtered_pricing:
            model_pricing = extract_model_pricing(filtered_pricing, model_id)

        pricing_cache[model_cache_key] = {
            "data": model_pricing,
            "timestamp": _utcnow(),
        }
        _pricing_cache_dirty = True
        return model_pricing

    except Exception as e:
        logger.error(f"Error fetching pricing for {model_id}: {e}")
        return pricing_cache.get(model_cache_key, {"error": "Pricing data unavailable"}).get(
            "data", {}
        )


def get_availale_models(client) -> dict[str, list[dict[str, Any]]]:
    response = client.list_foundation_models(
        byOutputModality="TEXT", byInferenceType="ON_DEMAND"
    )
    models = defaultdict(list)
    pricing_data, model_names = _get_pricing_catalog()

    for model in response["modelSummaries"]:
        model_arn = _get_model_identifier(model)
        if not model_arn:
            continue
        selected = model["providerName"].lower()
        provider_name = "mistral" if "mistral" in selected else selected
        response_streaming_supported = model["responseStreamingSupported"]
        model["modelName"] = model["modelName"].replace(".", ",")
        modelName = f"{model['providerName']} {model['modelName']}"

        model_id = model["modelId"]
        pricing_info = get_or_update_pricing(model_id, model_arn, pricing_data, model_names)
        if "error" in pricing_info:
            logger.warning(
                "Pricing unavailable for %s (%s). Registering the model anyway with empty pricing metadata.",
                modelName,
                model_id,
            )
            pricing_info = {model_id: {"input": {}, "output": {}, "cache_read_input": {}}}

        if response_streaming_supported:
            models[modelName].append(
                {
                    "model_arn": model_arn,
                    "provider_name": provider_name,
                    "response_streaming_supported": response_streaming_supported,
                    "pricing_info": pricing_info,
                    "model_id": model_id,
                }
            )
            assert "." not in modelName

    save_pricing_cache()
    return dict(models)


def get_available_guardrails(client):
    response = client.list_guardrails()
    guardrails = {"GUARDRAIL_0": "None"}
    index = 1
    for guardrail in response["guardrails"]:
        guardrail_details = client.list_guardrails(guardrailIdentifier=guardrail["id"])
        for detail in guardrail_details["guardrails"]:
            custom_identifier = (
                f'gr:{detail["name"]}:{detail["id"]}:v{detail["version"]}'
            )
            guardrails[f"GUARDRAIL_{index}"] = custom_identifier
            index += 1
    Guardrails = Enum("Guardrails", guardrails)
    return Guardrails


def get_class_name(name):
    class_name = re.sub(r"[^a-zA-Z0-9 \n.]", "", name.lower()).title()
    class_name = class_name.replace(" ", "").replace(",", "o")
    assert "." not in class_name
    return f"CustomBedrockLLM{class_name}"


def create_custom_bedrock_class(class_name, llm_info):
    runtime_model_arn = llm_info[0]["model_arn"]
    runtime_provider_name = llm_info[0]["provider_name"].lower()
    runtime_streaming_supported = llm_info[0]["response_streaming_supported"]

    class CustomBedrockLLM:
        @classmethod
        def default(cls, **kwargs):
            from langchain_aws import BedrockLLM, ChatBedrock

            RuntimeBaseClass = (
                BedrockLLM if runtime_provider_name in ("cohere",) else ChatBedrock
            )

            class RuntimeCustomBedrockLLM(RuntimeBaseClass):
                @staticmethod
                def _normalize_enum_value(value):
                    return value.value if isinstance(value, Enum) else value

                @classmethod
                def _normalize_bool(cls, value) -> bool:
                    value = cls._normalize_enum_value(value)
                    if isinstance(value, str):
                        return value.strip().lower() in {"1", "true", "yes", "on"}
                    return bool(value)

                def __init__(self, **runtime_kwargs):
                    input_kwargs = {
                        "model_id": runtime_model_arn,
                        "provider": runtime_provider_name,
                        "streaming": runtime_streaming_supported,
                        "model_kwargs": json.loads(runtime_kwargs.get("model_kwargs", "{}")),
                        "client": _get_bedrock_runtime_client(),
                    }
                    guardrail = self._normalize_enum_value(runtime_kwargs.get("guardrail_id", "None"))
                    if guardrail != "None":
                        parts = str(guardrail).split(":")
                        if len(parts) == 4:
                            _, _, guardrail_id, version = parts
                            input_kwargs["guardrails"] = {
                                "guardrailIdentifier": guardrail_id,
                                "guardrailVersion": version.replace("v", ""),
                                "trace": self._normalize_bool(runtime_kwargs.get("guardrail_trace", False)),
                            }
                        else:
                            logger.warning(
                                "Ignoring malformed Bedrock guardrail setting for %s: %s",
                                class_name,
                                guardrail,
                            )

                    input_kwargs = {
                        key: value for key, value in input_kwargs.items() if value is not None
                    }
                    super().__init__(**input_kwargs)

                    if runtime_kwargs.get("budget_mode", "Disabled") != "Disabled":
                        budget_limit = runtime_kwargs.get("budget_limit", "Unknown")
                        input_price = runtime_kwargs.get("input_token_price", "Unknown")
                        output_price = runtime_kwargs.get("output_token_price", "Unknown")
                        input_token_unit = runtime_kwargs.get("input_token_unit", "Unknown")
                        output_token_unit = runtime_kwargs.get("output_token_unit", "Unknown")

                        def parse_float(value, default=0.0):
                            if isinstance(value, (int, float)):
                                return float(value)
                            return (
                                float(value) if value.replace(".", "", 1).isdigit() else default
                            )

                        setattr(
                            RuntimeCustomBedrockLLM,
                            "_budget_config",
                            {
                                "budget_limit": parse_float(budget_limit),
                                "input_token_price": parse_float(input_price)
                                / parse_float(input_token_unit, default=1.0),
                                "output_token_price": parse_float(output_price)
                                / parse_float(output_token_unit, default=1.0),
                                "budget_mode": runtime_kwargs.get("budget_mode", "Disabled"),
                            },
                        )

                def get_current_model_cost(self):
                    return _load_current_model_cost()

                def compute_invocation_cost(self, input_tokens, output_tokens, total_tokens):
                    budget_config = getattr(self, "_budget_config", {})
                    input_price = budget_config.get("input_token_price", 0.0)
                    output_price = budget_config.get("output_token_price", 0.0)
                    current_request_cost = round(
                        (input_price * input_tokens) + (output_price * output_tokens), 6
                    )
                    model_total_cost = self.get_current_model_cost() + current_request_cost
                    _store_current_model_cost(model_total_cost)
                    return model_total_cost, current_request_cost

                def invoke(self, *args, **kwargs):
                    budget_config = getattr(self, "_budget_config", {})
                    budget_mode_value = self._normalize_enum_value(
                        budget_config.get("budget_mode", BudgetMode.DISABLED)
                    )
                    budget_mode = str(budget_mode_value)
                    budget_limit = float(budget_config.get("budget_limit", 0.0))
                    model_total_cost = self.get_current_model_cost()

                    alert_message = ""
                    if budget_limit > 0 and model_total_cost > budget_limit:
                        alert_message = (
                            f"⚠️ **Budget Limit Exceeded!** Budget Limit: ${budget_limit:.6f}"
                        )
                        logger.warning(alert_message)

                    if (
                        budget_mode == BudgetMode.BLOCK.value
                        and model_total_cost > budget_limit
                    ):
                        return AIMessage(
                            content="⛔ **Invocation Blocked Due to Budget Constraints.**\n"
                            "Your request cannot be processed because the total cost has exceeded the budget limit.\n"
                            "💰 **Cost Breakdown:**\n"
                            f"   - 🎯 **Budget Limit:** `${budget_limit:.6f}`\n"
                            f"   - 📊 Total Cost: `${model_total_cost:.6f}`"
                        )

                    if args:
                        normalized_input, dropped_messages = _normalize_bedrock_input(args[0])
                        if dropped_messages:
                            logger.warning(
                                "Dropped %s leading assistant message(s) before ChatBedrock.invoke to satisfy Bedrock Converse ordering.",
                                dropped_messages,
                            )
                        args = (normalized_input, *args[1:])
                    elif "input" in kwargs:
                        normalized_input, dropped_messages = _normalize_bedrock_input(kwargs["input"])
                        if dropped_messages:
                            logger.warning(
                                "Dropped %s leading assistant message(s) before ChatBedrock.invoke to satisfy Bedrock Converse ordering.",
                                dropped_messages,
                            )
                        kwargs["input"] = normalized_input

                    response = super().invoke(*args, **kwargs)

                    try:
                        usage_metadata = response.usage_metadata
                        model_total_cost, current_request_cost = self.compute_invocation_cost(
                            **usage_metadata
                        )
                        response.usage_metadata["current_request_cost"] = round(
                            current_request_cost, 6
                        )
                        response.usage_metadata["model_total_cost"] = round(
                            model_total_cost, 6
                        )

                        if budget_mode == BudgetMode.MONITOR.value:
                            logger.info(f"Invocation Cost: ${current_request_cost:.6f}")
                            logger.info(f"Total Cost (All Calls): ${model_total_cost:.6f}")
                            if alert_message:
                                logger.warning(alert_message)

                        if budget_mode == BudgetMode.NOTIFY.value and alert_message:
                            response.content += f"\n\n🚨 **{alert_message}** 🚨\n"

                        if budget_mode == BudgetMode.TRACE.value:
                            response.content += "\n\n"
                            if alert_message:
                                response.content += f"🚨 **{alert_message}** 🚨\n"
                            response.content += (
                                f"💰 **Cost Breakdown:**\n"
                                f"   - 📝 Request Cost: `${current_request_cost:.6f}`\n"
                                f"   - 📊 Total Cost: `${model_total_cost:.6f}`"
                            )

                        response.usage_metadata["budget_mode"] = budget_mode
                    except Exception as e:
                        logger.error(f"Error processing cost computation: {e}")

                    return response

            RuntimeCustomBedrockLLM.__name__ = class_name
            return RuntimeCustomBedrockLLM(**kwargs)

    CustomBedrockLLM.__name__ = class_name
    return CustomBedrockLLM


def get_model_price(llm_info):
    model_id = llm_info[0]["model_id"]
    pricing_info = llm_info[0].get("pricing_info", {})

    if isinstance(pricing_info, str):
        try:
            pricing_info = json.loads(pricing_info)
        except json.JSONDecodeError:
            print(
                f"Warning: Invalid pricing data format for {model_id}: {pricing_info}"
            )
            pricing_info = {}

    pricing_info = (
        pricing_info.get("rows", [pricing_info])[0]
        if isinstance(pricing_info, dict)
        else {}
    )

    input_token_price = pricing_info.get(model_id, {}).get("input", {})
    output_token_price = pricing_info.get(model_id, {}).get("output", {})

    return input_token_price, output_token_price


class BudgetMode(str, Enum):
    DISABLED = "Disabled"
    MONITOR = "Monitor"
    NOTIFY = "Notify"
    TRACE = "Trace"
    BLOCK = "Block"


def get_amazon_bedrock_llm_configs(
    amazon_llms: dict[str, list[dict[str, Any]]],
    Guardrails,
    config_llms: Optional[dict[str, Type[LLMSettings]]] = None,
) -> dict[str, Type[LLMSettings]]:
    if config_llms is None:
        config_llms = {}

    for model_name, llm_info in amazon_llms.items():
        class_name = get_class_name(model_name)
        custom_bedrock_class = create_custom_bedrock_class(class_name, llm_info)

        input_token_price, output_token_price = get_model_price(llm_info)

        input_price = (
            input_token_price.get("price", "0.0")
            if isinstance(input_token_price, dict)
            else None
        )
        input_price = (
            "Unknown" if input_price is None or input_price == 0.0 else input_price
        )

        output_price = (
            output_token_price.get("price", "0.0")
            if isinstance(output_token_price, dict)
            else None
        )
        output_price = (
            "Unknown" if output_price is None or output_price == 0.0 else output_price
        )

        input_currency = input_token_price.get("currency", "USD")
        output_currency = output_token_price.get("currency", "USD")
        input_unit = input_token_price.get("unit", "0.0")
        output_unit = output_token_price.get("unit", "0.0")

        class AmazonBedrockLLMConfig(LLMSettings):
            model_id: str = Field(
                default=llm_info[0]["model_arn"],
                description="The Amazon Resource Name (ARN) of the model.",
            )
            provider: str = Field(
                default=llm_info[0]["provider_name"],
                description="The name of the provider of the model.",
            )
            model_kwargs: Optional[str] = Field(
                default="{}",
                description="Additional keyword arguments for the model in JSON string format.",
            )
            guardrail_id: Any = Field(
                default=Guardrails.GUARDRAIL_0,
                description="The guardrail setting to be applied to the model.",
            )
            guardrail_trace: Optional[bool] = Field(
                default=False,
                description="A boolean indicating whether to trace guardrail execution.",
            )
            input_token_price: str = Field(
                default=input_price,
                description=f"The price per {input_unit} token (in {input_currency}).",
            )
            input_token_unit: str = Field(
                default=input_unit,
                description=f"The unit of the input token price.",
            )
            output_token_price: str = Field(
                default=output_price,
                description=f"The price per {output_unit} token (in {output_currency}).",
            )
            output_token_unit: str = Field(
                default=output_unit,
                description=f"The unit of the output token price.",
            )
            budget_mode: BudgetMode = Field(
                default=BudgetMode.DISABLED,
                description=(
                    "The budget mode for the model, which controls cost monitoring and enforcement. "
                    "Options:\n"
                    "Disabled: No budget tracking or restrictions.\n"
                    "Monitor: Logs the cost of each invocation without any notifications or enforcement.\n"
                    "Notify: Sends a warning notification when the budget limit is exceeded.\n"
                    "Trace: Appends cost breakdown details to the model’s response, including request cost and total usage.\n"
                    "Block: Prevents further invocations once the budget limit is exceeded, returning an error message instead."
                ),
            )
            budget_limit: Optional[str] = Field(
                default="",
                description="The maximum budget for the model.",
            )
            _pyclass: ClassVar[Type[Any]] = custom_bedrock_class
            model_config = ConfigDict(
                json_schema_extra={
                    "humanReadableName": f"Amazon Bedrock: {model_name}",
                    "description": f"Configuration for Amazon Bedrock LLMs ",
                    "link": "https://aws.amazon.com/bedrock/",
                },
                arbitrary_types_allowed=True,
                use_enum_values=True,
                validate_assignment=True,
                extra="allow",
            )

        new_class = type(class_name, (AmazonBedrockLLMConfig,), {})
        locals()[class_name] = new_class
        config_llms[model_name] = new_class
        assert "." not in class_name
        assert "." not in model_name
    return config_llms


def create_dynamic_model(amazon_llms: dict[str, list[dict[str, Any]]]) -> type[BaseModel]:
    dynamic_fields = {}
    default_model_name = None

    for model_name, llm_info in amazon_llms.items():
        if llm_info[0]["model_arn"].endswith(DEFAULT_MODEL):
            default_model_name = model_name
            break

    if default_model_name is None and amazon_llms:
        default_model_name = next(iter(amazon_llms))

    for model_name, llm_info in amazon_llms.items():
        model_name = model_name.replace(".", "o")
        dynamic_fields[model_name] = (
            bool,
            Field(
                default=model_name == (default_model_name or "").replace(".", "o"),
                description=f"Enable/disable the {model_name} model.",
            ),
        )
    dynamic_model = create_model(
        "DynamicModel",
        **dynamic_fields,
        __config__=ConfigDict(
            arbitrary_types_allowed=True,
            use_enum_values=True,
            validate_assignment=True,
            extra="allow",
        ),
    )
    return cast(type[BaseModel], dynamic_model)


_current_llms: list[Type[LLMSettings]] = []


def get_settings() -> type[BaseModel]:
    try:
        amazon_llms = get_availale_models(_get_bedrock_client())
    except Exception as e:
        logger.warning(
            "Live Bedrock model discovery failed, falling back to cached catalog so the plugin stays visible in Cheshire Cat. Error: %s",
            e,
        )
        amazon_llms = get_cached_available_models()

    if not amazon_llms:
        amazon_llms = get_seed_available_models()
        if amazon_llms:
            logger.warning(
                "Bedrock live discovery and cached catalog are unavailable; falling back to the seeded model catalog to keep the plugin visible in Cheshire Cat."
            )

    try:
        Guardrails = get_available_guardrails(_get_bedrock_client())
    except Exception as e:
        logger.warning(f"Unable to load Bedrock guardrails, defaulting to None only: {e}")
        Guardrails = Enum("Guardrails", {"GUARDRAIL_0": "None"})

    config_llms = get_amazon_bedrock_llm_configs(amazon_llms, Guardrails)
    DynamicModel = create_dynamic_model(amazon_llms)

    class AmazonBedrockLLMSettings(DynamicModel):
        model_config = ConfigDict(
            arbitrary_types_allowed=True,
            use_enum_values=True,
            validate_assignment=True,
            extra="allow",
        )

        def init_llm(self):
            global _current_llms
            if not _current_llms:
                _current_llms = []

        def get_llms(self):
            global _current_llms
            return _current_llms

        @model_validator(mode="before")
        def validate(cls, values):
            global _current_llms
            _current_llms = []
            for llm in values.keys():
                if llm in values and values[llm]:
                    _current_llms.append(config_llms[llm])
            log.info("Dynamically Selected LLMs:")
            log.info(
                [
                    llm.model_config["json_schema_extra"]["humanReadableName"]
                    for llm in _current_llms
                ]
            )
            return values

    return AmazonBedrockLLMSettings


@plugin
def settings_model():
    logger.info("Amazon Bedrock settings_model requested by Cheshire Cat.")
    return get_settings()


def factory_pipeline():
    AmazonBedrockLLMSettings = get_settings()
    default_settings = cast(Any, AmazonBedrockLLMSettings()).model_dump()
    plugin_settings = _load_plugin_settings()
    effective_settings = {**default_settings, **plugin_settings}
    settings = cast(Any, AmazonBedrockLLMSettings)(**effective_settings)
    llms = settings.get_llms()
    logger.info("Registering %s Amazon Bedrock LLM configuration(s) in Cheshire Cat.", len(llms))
    return llms


@plugin
def activated(plugin):
    _ensure_llm_factory_patch()
    logger.info("Amazon Bedrock plugin activated.")


@hook
def agent_prompt_prefix(prefix, cat):
    prefix = (
        "Please do not include any cost breakdowns, request costs, or total cost information in responses. "
        "Focus only on the main conversation topic and user requests."
    )
    return prefix


@tool(
    "Reset Cumulative Model Cost",
    return_direct=False,
    examples=[
        "Reset the stored model cost data.",
        "Clear all accumulated cost records for the model.",
        "Delete the current cost cache and start fresh.",
    ],
)
def reset_cached_model_costs(data, cat):
    """Reset the cumulative model cost data.

    This function clears the cached record of the total cost accumulated across all model invocations.
    It ensures that future cost tracking starts from zero.
    """
    try:
        _store_current_model_cost(0.0)
        return "✅ Cumulative model cost has been reset."
    except Exception as e:
        return f"❌ Error resetting cumulative model cost: {str(e)}"


# @tool(
#     "Reset Model Pricing Data",
#     return_direct=False,
#     examples=[
#         "Reset the token cost statistics for the model.",
#         "Clear all stored pricing information for LLM calls.",
#         "Delete the pricing cache and refresh it.",
#     ],
# )
# def reset_cached_model_pricing(data, cat):
#     """Reset token cost statistics for each LLM call.

#     This function clears the cached pricing data, which tracks the cost per token for different LLM calls.
#     After resetting, the system will need to reload or re-fetch pricing information.
#     """
#     try:
#         with open(CACHED_PRICING_FILE, "w") as f:
#             json.dump({}, f)
#         return "✅ Token cost statistics have been reset."
#     except Exception as e:
#         return f"❌ Error resetting token cost statistics: {str(e)}"


@tool(
    "Get Current Model Cost",
    return_direct=False,
    examples=[
        "What is the total cost of my model usage?",
        "Show me the current cumulative cost for the model.",
        "How much have I spent on LLM calls so far?",
    ],
)
def get_current_model_cost(data, cat):
    """Retrieve the current cumulative model cost.

    Reads the cached model cost data and returns the total cost accumulated across all model invocations.
    """
    try:
        total_cost = _load_current_model_cost()

        return (
            f"💰 **Total Accumulated Cost:** `${total_cost:.6f}`\n"
            "🔹 This includes all previous model invocations.\n"
            "⚠️ *The cost of the current request is not included and will be added after execution.*"
        )
    except Exception as e:
        return f"❌ Error retrieving model cost: {str(e)}"


@tool(
    "Get Current Model Pricing",
    return_direct=False,
    examples=[
        "How much does my current model charge per token?",
        "What is the pricing for each model call?",
        "Show the token price for my LLM model.",
    ],
)
def get_current_model_pricing(data, cat):
    """Retrieve the pricing information for the current model.

    Reads the cached pricing data and returns the cost per token for the model in use.
    """
    try:
        model_class_name = crud.get_setting_by_name("llm_selected")["value"]["name"]
        model_arn = crud.get_setting_by_name(model_class_name)["value"]["model_id"]
        model_id = model_arn.split("/")[-1]

        if not os.path.exists(CACHED_PRICING_FILE):
            return "⚠️ No pricing data found. The cache might be empty."

        with open(CACHED_PRICING_FILE, "r") as f:
            pricing_data = json.load(f)

        model_pricing = pricing_data.get(model_arn, {}).get("data", {}).get(model_id)
        if not model_pricing:
            return f"⚠️ No pricing information available for model `{model_arn}`."

        input_price = model_pricing.get("input", {}).get("price")
        input_unit = model_pricing.get("input", {}).get("unit", 1000)
        output_price = model_pricing.get("output", {}).get("price")
        output_unit = model_pricing.get("output", {}).get("unit", 1000)

        input_price_str = (
            f"${float(input_price):.6f}"
            if isinstance(input_price, (int, float))
            else "N/A"
        )
        output_price_str = (
            f"${float(output_price):.6f}"
            if isinstance(output_price, (int, float))
            else "N/A"
        )

        return (
            f"💲 **Current Model Pricing for `{model_arn}`**\n"
            f"🔹 **Input Cost:** {input_price_str} per {input_unit} tokens\n"
            f"🔹 **Output Cost:** {output_price_str} per {output_unit} tokens"
        )

    except Exception as e:
        return f"❌ **Error retrieving model pricing:** {str(e)}"


@tool(
    "Get Current Model",
    return_direct=False,
    examples=[
        "Which LLM model am I using?",
        "What is my current AI model?",
        "Show the model name I am working with.",
    ],
)
def get_current_model(data, cat):
    """Retrieve the currently selected AI model.

    Returns the name and ARN of the model in use.
    """
    try:
        model_class_name = crud.get_setting_by_name("llm_selected")["value"]["name"]
        model_arn = crud.get_setting_by_name(model_class_name)["value"]["model_id"]

        return (
            f"🤖 **Current Model Information**\n"
            f"🔹 **Model Class Name:** `{model_class_name}`\n"
            f"🔹 **Model Amazon Resource Name:** `{model_arn}`"
            "You can use this information to identify the model and its capabilities."
        )

    except Exception as e:
        return f"❌ **Error retrieving model information:** {str(e)}"


@hook
def factory_allowed_llms(allowed, cat) -> List:
    logger.info(
        "Amazon Bedrock factory_allowed_llms invoked with %s existing Cheshire Cat LLM configuration(s).",
        len(allowed),
    )
    return allowed + factory_pipeline()


_ensure_llm_factory_patch()
logger.info("Amazon Bedrock LLM plugin module initialization completed.")


