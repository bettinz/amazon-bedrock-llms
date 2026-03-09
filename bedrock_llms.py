import os
import json
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Type

from pydantic import Field, ConfigDict
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
import logging

from cat.mad_hatter.decorators import tool, hook, plugin
from cat.mad_hatter.mad_hatter import MadHatter
from cat.factory.llm import LLMSettings
from cat.plugins.aws_integration import Boto3

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PLUGIN_NAME = "amazon_bedrock_llms"

# ── Nova model IDs ────────────────────────────────────────────────────────────
NOVA_PRO_MODEL_ID    = "amazon.nova-pro-v1:0"
NOVA_LITE_MODEL_ID   = "amazon.nova-lite-v1:0"
NOVA_2_LITE_MODEL_ID = "amazon.nova-2-lite-v1:0"

DEFAULT_MODEL_ID = NOVA_PRO_MODEL_ID

# ── Claude model IDs ──────────────────────────────────────────────────────────
CLAUDE_SONNET_4_6_MODEL_ID = "anthropic.claude-sonnet-4-6"
CLAUDE_SONNET_4_5_MODEL_ID = "anthropic.claude-sonnet-4-5-20250929-v1:0"
CLAUDE_SONNET_4_MODEL_ID   = "anthropic.claude-sonnet-4-20250514-v1:0"
CLAUDE_HAIKU_4_5_MODEL_ID  = "anthropic.claude-haiku-4-5-20251001-v1:0"
CLAUDE_OPUS_4_5_MODEL_ID   = "anthropic.claude-opus-4-5-20251101-v1:0"
CLAUDE_OPUS_4_6_MODEL_ID   = "anthropic.claude-opus-4-6-v1"

DEFAULT_CLAUDE_MODEL_ID = CLAUDE_SONNET_4_6_MODEL_ID

# ── Pricing (per 1K tokens, US East) ─────────────────────────────────────────
NOVA_PRO_INPUT_PRICE     = 0.0008
NOVA_PRO_OUTPUT_PRICE    = 0.0032
NOVA_LITE_INPUT_PRICE    = 0.00006
NOVA_LITE_OUTPUT_PRICE   = 0.00024
NOVA_2_LITE_INPUT_PRICE  = 0.00006
NOVA_2_LITE_OUTPUT_PRICE = 0.00024
NOVA_TOKEN_UNIT          = 1000

CLAUDE_SONNET_4_INPUT_PRICE  = 0.003
CLAUDE_SONNET_4_OUTPUT_PRICE = 0.015
CLAUDE_HAIKU_4_INPUT_PRICE   = 0.0008
CLAUDE_HAIKU_4_OUTPUT_PRICE  = 0.004
CLAUDE_OPUS_4_INPUT_PRICE    = 0.015
CLAUDE_OPUS_4_OUTPUT_PRICE   = 0.075
CLAUDE_TOKEN_UNIT            = 1000


def get_cached_cost_file_path():
    return os.path.join(
        MadHatter().plugins.get(PLUGIN_NAME)._path, "cached_model_costs.json"
    )


class BudgetMode(str, Enum):
    DISABLED = "Disabled"
    MONITOR  = "Monitor"
    NOTIFY   = "Notify"
    TRACE    = "Trace"
    BLOCK    = "Block"


def get_default_pricing(model_id: str):
    m = model_id.lower()
    if "nova-2" in m and "lite" in m:
        return NOVA_2_LITE_INPUT_PRICE, NOVA_2_LITE_OUTPUT_PRICE
    if "lite" in m:
        return NOVA_LITE_INPUT_PRICE, NOVA_LITE_OUTPUT_PRICE
    return NOVA_PRO_INPUT_PRICE, NOVA_PRO_OUTPUT_PRICE


def get_default_claude_pricing(model_id: str):
    m = model_id.lower()
    if "haiku" in m:
        return CLAUDE_HAIKU_4_INPUT_PRICE, CLAUDE_HAIKU_4_OUTPUT_PRICE
    if "opus" in m:
        return CLAUDE_OPUS_4_INPUT_PRICE, CLAUDE_OPUS_4_OUTPUT_PRICE
    return CLAUDE_SONNET_4_INPUT_PRICE, CLAUDE_SONNET_4_OUTPUT_PRICE


def _extract_text(content) -> str:
    """Extract plain text from a message content that may be a string or a list of content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", str(block)))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def _messages_to_bedrock(messages: List[BaseMessage]):
    """Convert LangChain messages to Bedrock Converse API format."""
    system_parts = []
    converse_messages = []

    for msg in messages:
        text = _extract_text(msg.content)
        if isinstance(msg, SystemMessage):
            system_parts.append({"text": text})
        elif isinstance(msg, HumanMessage):
            converse_messages.append({"role": "user", "content": [{"text": text}]})
        elif isinstance(msg, AIMessage):
            converse_messages.append({"role": "assistant", "content": [{"text": text}]})
        else:
            converse_messages.append({"role": "user", "content": [{"text": text}]})

    return system_parts, converse_messages


def _load_cost_cache(path: str) -> float:
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return float((json.load(f) or {}).get("current_cost", 0.0))
        except (json.JSONDecodeError, IOError):
            pass
    return 0.0


def _save_cost_cache(path: str, total: float):
    try:
        with open(path, "w") as f:
            json.dump({"current_cost": total}, f, indent=4)
    except IOError as e:
        logger.error(f"Error saving pricing cache: {e}")


class BedrockConverseChat(BaseChatModel):
    """
    LangChain BaseChatModel that calls AWS Bedrock Converse API directly via boto3.
    No dependency on langchain-aws.
    """
    model_id: str
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 0.9
    token_unit: int = 1000
    input_token_price: float = 0.0
    output_token_price: float = 0.0
    budget_mode: str = "Disabled"
    budget_limit: float = 0.0

    # cached boto3 client — excluded from pydantic serialization
    _client: Any = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        """Create and cache the boto3 bedrock-runtime client once at init time."""
        object.__setattr__(self, "_client", Boto3().get_client("bedrock-runtime"))

    @property
    def _llm_type(self) -> str:
        return "bedrock-converse"

    def _get_client(self):
        if self._client is None:
            object.__setattr__(self, "_client", Boto3().get_client("bedrock-runtime"))
        return self._client

    def _build_inference_config(self) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {}
        if self.max_tokens:
            cfg["maxTokens"] = self.max_tokens
        if self.temperature is not None:
            cfg["temperature"] = self.temperature
        if self.top_p is not None:
            cfg["topP"] = self.top_p
        return cfg

    def _get_current_cost(self) -> float:
        return _load_cost_cache(get_cached_cost_file_path())

    def _compute_and_save_cost(self, input_tokens: int, output_tokens: int):
        price_in  = (self.input_token_price  / self.token_unit) * input_tokens
        price_out = (self.output_token_price / self.token_unit) * output_tokens
        request_cost = round(price_in + price_out, 6)
        total_cost   = round(self._get_current_cost() + request_cost, 6)
        _save_cost_cache(get_cached_cost_file_path(), total_cost)
        return total_cost, request_cost

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> ChatResult:
        budget_mode  = self.budget_mode.capitalize()
        budget_limit = self.budget_limit
        total_cost   = self._get_current_cost()

        alert_message = ""
        if budget_limit > 0 and total_cost > budget_limit:
            alert_message = f"⚠️ **Budget Limit Exceeded!** Budget Limit: ${budget_limit:.6f}"
            logger.warning(alert_message)

        if budget_mode == BudgetMode.BLOCK.value and budget_limit > 0 and total_cost > budget_limit:
            return ChatResult(generations=[ChatGeneration(message=AIMessage(
                content=(
                    "⛔ **Invocation Blocked Due to Budget Constraints.**\n"
                    f"   - 🎯 **Budget Limit:** `${budget_limit:.6f}`\n"
                    f"   - 📊 Total Cost: `${total_cost:.6f}`"
                )
            ))])

        system_parts, converse_messages = _messages_to_bedrock(messages)

        request: Dict[str, Any] = {
            "modelId": self.model_id,
            "messages": converse_messages,
            "inferenceConfig": self._build_inference_config(),
        }
        if system_parts:
            request["system"] = system_parts

        client = self._get_client()
        response = client.converse(**request)

        output_content = response["output"]["message"]["content"][0]["text"]
        usage = response.get("usage", {})
        input_tokens  = usage.get("inputTokens", 0)
        output_tokens = usage.get("outputTokens", 0)

        total_cost, request_cost = self._compute_and_save_cost(input_tokens, output_tokens)

        if budget_mode == BudgetMode.MONITOR.value:
            logger.info(f"Invocation Cost: ${request_cost:.6f}")
            logger.info(f"Total Cost (All Calls): ${total_cost:.6f}")
            if alert_message:
                logger.warning(alert_message)

        if budget_mode == BudgetMode.NOTIFY.value and alert_message:
            output_content += f"\n\n🚨 **{alert_message}** 🚨\n"

        if budget_mode == BudgetMode.TRACE.value:
            output_content += "\n\n"
            if alert_message:
                output_content += f"🚨 **{alert_message}** 🚨\n"
            output_content += (
                f"💰 **Cost Breakdown:**\n"
                f"   - 📝 Request Cost: `${request_cost:.6f}`\n"
                f"   - 📊 Total Cost: `${total_cost:.6f}`"
            )

        ai_message = AIMessage(
            content=output_content,
            usage_metadata={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "current_request_cost": request_cost,
                "model_total_cost": total_cost,
                "budget_mode": budget_mode,
            },
        )
        return ChatResult(generations=[ChatGeneration(message=ai_message)])


# ── Nova LLM ──────────────────────────────────────────────────────────────────

class NovaLLM(BedrockConverseChat):
    """Amazon Nova models via AWS Bedrock Converse API."""

    def __init__(self, **kwargs):
        model_id = kwargs.get("model_id", DEFAULT_MODEL_ID)
        default_in, default_out = get_default_pricing(model_id)

        super().__init__(
            model_id=model_id,
            temperature=float(kwargs.get("temperature", 0.7)),
            max_tokens=int(kwargs.get("max_tokens", 4096)),
            top_p=float(kwargs.get("top_p", 0.9)),
            token_unit=NOVA_TOKEN_UNIT,
            input_token_price=float(kwargs.get("input_token_price", default_in)),
            output_token_price=float(kwargs.get("output_token_price", default_out)),
            budget_mode=kwargs.get("budget_mode", "Disabled"),
            budget_limit=float(kwargs.get("budget_limit", 0.0)),
        )


class NovaLLMConfig(LLMSettings):
    """Configuration for Amazon Nova LLMs."""

    model_id: Literal[
        "amazon.nova-pro-v1:0",
        "amazon.nova-lite-v1:0",
        "amazon.nova-2-lite-v1:0",
    ] = Field(default=DEFAULT_MODEL_ID, description="Amazon Nova model ID.")

    temperature: float = Field(default=0.7, ge=0.0, le=1.0,
        description="Randomness (0=deterministic, 1=creative).")
    max_tokens: int = Field(default=4096, ge=1, le=100000,
        description="Maximum tokens to generate.")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0,
        description="Nucleus sampling threshold.")
    input_token_price: float = Field(default=NOVA_PRO_INPUT_PRICE,
        description=f"Price per {NOVA_TOKEN_UNIT} input tokens (USD).")
    output_token_price: float = Field(default=NOVA_PRO_OUTPUT_PRICE,
        description=f"Price per {NOVA_TOKEN_UNIT} output tokens (USD).")
    budget_mode: BudgetMode = Field(default=BudgetMode.DISABLED,
        description="Budget enforcement mode.")
    budget_limit: Optional[float] = Field(default=0.0,
        description="Maximum budget in USD (0 = unlimited).")

    _pyclass: Type = NovaLLM

    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "Amazon Nova",
            "description": "Amazon Nova models (Pro, Lite, Nova 2 Lite) via AWS Bedrock",
            "link": "https://aws.amazon.com/bedrock/",
        },
        arbitrary_types_allowed=True,
        use_enum_values=True,
        validate_assignment=True,
        extra="allow",
    )


# ── Claude LLM ────────────────────────────────────────────────────────────────

class ClaudeLLM(BedrockConverseChat):
    """Anthropic Claude models via AWS Bedrock Converse API."""

    def __init__(self, **kwargs):
        model_id = kwargs.get("model_id", DEFAULT_CLAUDE_MODEL_ID)
        default_in, default_out = get_default_claude_pricing(model_id)

        super().__init__(
            model_id=model_id,
            temperature=float(kwargs.get("temperature", 0.7)),
            max_tokens=int(kwargs.get("max_tokens", 4096)),
            top_p=float(kwargs.get("top_p", 0.9)),
            token_unit=CLAUDE_TOKEN_UNIT,
            input_token_price=float(kwargs.get("input_token_price", default_in)),
            output_token_price=float(kwargs.get("output_token_price", default_out)),
            budget_mode=kwargs.get("budget_mode", "Disabled"),
            budget_limit=float(kwargs.get("budget_limit", 0.0)),
        )


class ClaudeLLMConfig(LLMSettings):
    """Configuration for Anthropic Claude models via AWS Bedrock."""

    model_id: Literal[
        "anthropic.claude-sonnet-4-6",
        "anthropic.claude-sonnet-4-5-20250929-v1:0",
        "anthropic.claude-sonnet-4-20250514-v1:0",
        "anthropic.claude-haiku-4-5-20251001-v1:0",
        "anthropic.claude-opus-4-6-v1",
        "anthropic.claude-opus-4-5-20251101-v1:0",
    ] = Field(default=DEFAULT_CLAUDE_MODEL_ID, description="Anthropic Claude model ID on AWS Bedrock.")

    temperature: float = Field(default=0.7, ge=0.0, le=1.0,
        description="Randomness (0=deterministic, 1=creative).")
    max_tokens: int = Field(default=4096, ge=1, le=200000,
        description="Maximum tokens to generate.")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0,
        description="Nucleus sampling threshold.")
    input_token_price: float = Field(default=CLAUDE_SONNET_4_INPUT_PRICE,
        description=f"Price per {CLAUDE_TOKEN_UNIT} input tokens (USD).")
    output_token_price: float = Field(default=CLAUDE_SONNET_4_OUTPUT_PRICE,
        description=f"Price per {CLAUDE_TOKEN_UNIT} output tokens (USD).")
    budget_mode: BudgetMode = Field(default=BudgetMode.DISABLED,
        description="Budget enforcement mode.")
    budget_limit: Optional[float] = Field(default=0.0,
        description="Maximum budget in USD (0 = unlimited).")

    _pyclass: Type = ClaudeLLM

    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "Anthropic Claude (Bedrock)",
            "description": "Anthropic Claude models (Sonnet 4.6, Haiku 4.5, Opus 4.6) via AWS Bedrock",
            "link": "https://aws.amazon.com/bedrock/claude/",
        },
        arbitrary_types_allowed=True,
        use_enum_values=True,
        validate_assignment=True,
        extra="allow",
    )


# ── Hooks & Tools ─────────────────────────────────────────────────────────────

@hook
def agent_prompt_prefix(prefix, cat):
    return prefix + "\nPlease do not include any cost breakdowns, request costs, or total cost information in responses."


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
    """Reset the cumulative model cost data."""
    try:
        with open(get_cached_cost_file_path(), "w") as f:
            json.dump({}, f)
        return "✅ Cumulative model cost has been reset."
    except Exception as e:
        return f"❌ Error resetting cumulative model cost: {str(e)}"


@tool(
    "Get Current Model Cost",
    return_direct=False,
    examples=[
        "What is the total cost of my model usage?",
        "How much have I spent on LLM calls so far?",
    ],
)
def get_current_model_cost(data, cat):
    """Retrieve the current cumulative model cost."""
    try:
        path = get_cached_cost_file_path()
        if not os.path.exists(path):
            return "⚠️ No cost data found."
        total_cost = _load_cost_cache(path)
        return (
            f"💰 **Total Accumulated Cost:** `${total_cost:.6f}`\n"
            "🔹 This includes all previous model invocations.\n"
            "⚠️ *The cost of the current request is not included.*"
        )
    except Exception as e:
        return f"❌ Error retrieving model cost: {str(e)}"


@tool(
    "Get Current Model Pricing",
    return_direct=False,
    examples=[
        "How much does Nova Pro charge per token?",
        "What is the pricing for Claude Sonnet?",
    ],
)
def get_current_model_pricing(data, cat):
    """Retrieve pricing information for all supported models."""
    return (
        f"💲 **Amazon Nova Pricing**\n\n"
        f"**Nova Pro:** ${NOVA_PRO_INPUT_PRICE:.6f} in / ${NOVA_PRO_OUTPUT_PRICE:.6f} out per {NOVA_TOKEN_UNIT} tokens\n"
        f"**Nova Lite:** ${NOVA_LITE_INPUT_PRICE:.6f} in / ${NOVA_LITE_OUTPUT_PRICE:.6f} out per {NOVA_TOKEN_UNIT} tokens\n"
        f"**Nova 2 Lite:** ${NOVA_2_LITE_INPUT_PRICE:.6f} in / ${NOVA_2_LITE_OUTPUT_PRICE:.6f} out per {NOVA_TOKEN_UNIT} tokens\n\n"
        f"💲 **Anthropic Claude Pricing**\n\n"
        f"**Claude Sonnet 4.x:** ${CLAUDE_SONNET_4_INPUT_PRICE:.6f} in / ${CLAUDE_SONNET_4_OUTPUT_PRICE:.6f} out per {CLAUDE_TOKEN_UNIT} tokens\n"
        f"**Claude Haiku 4.x:** ${CLAUDE_HAIKU_4_INPUT_PRICE:.6f} in / ${CLAUDE_HAIKU_4_OUTPUT_PRICE:.6f} out per {CLAUDE_TOKEN_UNIT} tokens\n"
        f"**Claude Opus 4.x:** ${CLAUDE_OPUS_4_INPUT_PRICE:.6f} in / ${CLAUDE_OPUS_4_OUTPUT_PRICE:.6f} out per {CLAUDE_TOKEN_UNIT} tokens"
    )


@tool(
    "Get Current Model",
    return_direct=False,
    examples=[
        "Which LLM model am I using?",
        "What models are available?",
    ],
)
def get_current_model(data, cat):
    """Retrieve available AI models."""
    return (
        f"🤖 **Available Models on AWS Bedrock**\n\n"
        f"**Amazon Nova:**\n"
        f"   - Nova Pro: `{NOVA_PRO_MODEL_ID}`\n"
        f"   - Nova Lite: `{NOVA_LITE_MODEL_ID}`\n"
        f"   - Nova 2 Lite: `{NOVA_2_LITE_MODEL_ID}`\n\n"
        f"**Anthropic Claude:**\n"
        f"   - Claude Sonnet 4.6: `{CLAUDE_SONNET_4_6_MODEL_ID}`\n"
        f"   - Claude Sonnet 4.5: `{CLAUDE_SONNET_4_5_MODEL_ID}`\n"
        f"   - Claude Sonnet 4: `{CLAUDE_SONNET_4_MODEL_ID}`\n"
        f"   - Claude Haiku 4.5: `{CLAUDE_HAIKU_4_5_MODEL_ID}`\n"
        f"   - Claude Opus 4.6: `{CLAUDE_OPUS_4_6_MODEL_ID}`\n"
        f"   - Claude Opus 4.5: `{CLAUDE_OPUS_4_5_MODEL_ID}`\n"
    )


@hook
def factory_allowed_llms(allowed, cat) -> List:
    return allowed + [NovaLLMConfig, ClaudeLLMConfig]


@plugin
def settings_model():
    from pydantic import BaseModel
    class NovaPluginSettings(BaseModel):
        """Settings for Amazon Bedrock LLM plugin."""
        pass

    return NovaPluginSettings
