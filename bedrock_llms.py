import os
import json
from enum import Enum
from typing import List, Optional, Type

from pydantic import Field, ConfigDict
from langchain_core.messages import AIMessage
from langchain_aws import ChatBedrockConverse
import logging

from cat.mad_hatter.decorators import tool, hook, plugin
from cat.mad_hatter.mad_hatter import MadHatter
from cat.factory.llm import LLMSettings
from cat.plugins.aws_integration import Boto3

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# IMPORTANT: This must match the folder name of the plugin
PLUGIN_NAME = "amazon_bedrock_llms"

# Default Nova model IDs
# Model ID format: amazon.nova-{variant}-v{version}:0
# The region is determined by the AWS client configuration, NOT by a prefix in the model ID

# Nova v1 models
NOVA_PRO_MODEL_ID = "amazon.nova-pro-v1:0"
NOVA_LITE_MODEL_ID = "amazon.nova-lite-v1:0"

# Nova 2 models (next-gen)
NOVA_2_LITE_MODEL_ID = "amazon.nova-2-lite-v1:0"

DEFAULT_MODEL_ID = NOVA_PRO_MODEL_ID


def get_cached_cost_file_path():
    """Get the path to the cached cost file lazily."""
    return os.path.join(
        MadHatter().plugins.get(PLUGIN_NAME)._path, "cached_model_costs.json"
    )


class BudgetMode(str, Enum):
    DISABLED = "Disabled"
    MONITOR = "Monitor"
    NOTIFY = "Notify"
    TRACE = "Trace"
    BLOCK = "Block"


# Nova Pro v1 pricing (per 1K tokens) - US East region
NOVA_PRO_INPUT_PRICE = 0.0008  # $0.0008 per 1K input tokens
NOVA_PRO_OUTPUT_PRICE = 0.0032  # $0.0032 per 1K output tokens

# Nova Lite v1 pricing (per 1K tokens) - US East region
NOVA_LITE_INPUT_PRICE = 0.00006  # $0.00006 per 1K input tokens
NOVA_LITE_OUTPUT_PRICE = 0.00024  # $0.00024 per 1K output tokens

# Nova 2 Lite pricing (per 1K tokens) - US East region
NOVA_2_LITE_INPUT_PRICE = 0.00006   # $0.00006 per 1K input tokens
NOVA_2_LITE_OUTPUT_PRICE = 0.00024  # $0.00024 per 1K output tokens

NOVA_TOKEN_UNIT = 1000

# ── Claude model IDs ──────────────────────────────────────────────────────────
CLAUDE_SONNET_4_6_MODEL_ID = "anthropic.claude-sonnet-4-6"
CLAUDE_SONNET_4_5_MODEL_ID = "anthropic.claude-sonnet-4-5-20250929-v1:0"
CLAUDE_SONNET_4_MODEL_ID   = "anthropic.claude-sonnet-4-20250514-v1:0"
CLAUDE_HAIKU_4_5_MODEL_ID  = "anthropic.claude-haiku-4-5-20251001-v1:0"
CLAUDE_OPUS_4_5_MODEL_ID   = "anthropic.claude-opus-4-5-20251101-v1:0"
CLAUDE_OPUS_4_6_MODEL_ID   = "anthropic.claude-opus-4-6-v1"

DEFAULT_CLAUDE_MODEL_ID = CLAUDE_SONNET_4_6_MODEL_ID

# Claude Sonnet 4.x pricing (per 1K tokens) - US East region
CLAUDE_SONNET_4_INPUT_PRICE  = 0.003   # $0.003 per 1K input tokens
CLAUDE_SONNET_4_OUTPUT_PRICE = 0.015   # $0.015 per 1K output tokens

# Claude Haiku 4.x pricing (per 1K tokens) - US East region
CLAUDE_HAIKU_4_INPUT_PRICE  = 0.0008   # $0.0008 per 1K input tokens
CLAUDE_HAIKU_4_OUTPUT_PRICE = 0.004    # $0.004 per 1K output tokens

# Claude Opus 4.x pricing (per 1K tokens) - US East region
CLAUDE_OPUS_4_INPUT_PRICE  = 0.015    # $0.015 per 1K input tokens
CLAUDE_OPUS_4_OUTPUT_PRICE = 0.075    # $0.075 per 1K output tokens

CLAUDE_TOKEN_UNIT = 1000


def get_default_claude_pricing(model_id: str):
    """Returns default pricing based on Claude model ID."""
    m = model_id.lower()
    if "haiku" in m:
        return CLAUDE_HAIKU_4_INPUT_PRICE, CLAUDE_HAIKU_4_OUTPUT_PRICE
    if "opus" in m:
        return CLAUDE_OPUS_4_INPUT_PRICE, CLAUDE_OPUS_4_OUTPUT_PRICE
    return CLAUDE_SONNET_4_INPUT_PRICE, CLAUDE_SONNET_4_OUTPUT_PRICE


def get_default_pricing(model_id: str):
    """Returns default pricing based on model ID."""
    model_id_lower = model_id.lower()
    is_nova2 = "nova-2" in model_id_lower
    is_lite = "lite" in model_id_lower

    if is_nova2 and is_lite:
        return NOVA_2_LITE_INPUT_PRICE, NOVA_2_LITE_OUTPUT_PRICE
    if is_lite:
        return NOVA_LITE_INPUT_PRICE, NOVA_LITE_OUTPUT_PRICE
    return NOVA_PRO_INPUT_PRICE, NOVA_PRO_OUTPUT_PRICE


class NovaLLM(ChatBedrockConverse):
    """Custom ChatBedrockConverse class for Amazon Nova models (Pro and Lite)."""

    def __init__(self, **kwargs):
        model_id = kwargs.get("model_id", DEFAULT_MODEL_ID)
        default_input_price, default_output_price = get_default_pricing(model_id)

        # Parse model_kwargs if it's a string
        model_kwargs = kwargs.get("model_kwargs", "{}")
        if isinstance(model_kwargs, str):
            try:
                model_kwargs = json.loads(model_kwargs)
            except json.JSONDecodeError:
                model_kwargs = {}

        # Get inference parameters from settings
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 4096)
        top_p = kwargs.get("top_p", 0.9)

        # Build input kwargs for ChatBedrockConverse
        input_kwargs = {
            "model": model_id,
            "client": Boto3().get_client("bedrock-runtime"),
            "temperature": float(temperature) if temperature is not None else None,
            "max_tokens": int(max_tokens) if max_tokens is not None else None,
            "top_p": float(top_p) if top_p is not None else None,
        }

        if model_kwargs:
            input_kwargs["additional_model_request_fields"] = model_kwargs

        # Remove None values
        input_kwargs = {k: v for k, v in input_kwargs.items() if v is not None}

        super(NovaLLM, self).__init__(**input_kwargs)

        if kwargs.get("budget_mode", "Disabled") != "Disabled":
            budget_limit = kwargs.get("budget_limit", 0.0)
            input_price = kwargs.get("input_token_price", default_input_price)
            output_price = kwargs.get("output_token_price", default_output_price)

            def parse_float(value, default=0.0):
                if isinstance(value, (int, float)):
                    return float(value)
                return float(value) if str(value).replace(".", "", 1).isdigit() else default

            budget_limit = parse_float(budget_limit)
            input_price = parse_float(input_price)
            output_price = parse_float(output_price)

            setattr(
                NovaLLM,
                "_budget_config",
                {
                    "budget_limit": budget_limit,
                    "input_token_price": input_price / NOVA_TOKEN_UNIT,
                    "output_token_price": output_price / NOVA_TOKEN_UNIT,
                    "budget_mode": kwargs.get("budget_mode", "Disabled"),
                },
            )

    def get_current_model_cost(self):
        """Retrieves the total model cost from the cache file."""
        cached_cost_file = get_cached_cost_file_path()
        if os.path.exists(cached_cost_file):
            try:
                with open(cached_cost_file, "r") as file:
                    pricing_cache = json.load(file) or {}
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load pricing cache. Error: {e}")
                pricing_cache = {}
        else:
            pricing_cache = {}

        return float(pricing_cache.get("current_cost", 0.0))

    def compute_invocation_cost(self, input_tokens, output_tokens, total_tokens):
        """Computes cost for the current request and updates the total model cost."""
        budget_config = getattr(self, "_budget_config", {})
        input_price = budget_config.get("input_token_price", 0.0)
        output_price = budget_config.get("output_token_price", 0.0)

        input_cost = input_price * input_tokens
        output_cost = output_price * output_tokens
        current_request_cost = round(input_cost + output_cost, 6)

        model_total_cost = self.get_current_model_cost() + current_request_cost

        pricing_cache = {"current_cost": model_total_cost}
        try:
            cached_cost_file = get_cached_cost_file_path()
            with open(cached_cost_file, "w") as file:
                json.dump(pricing_cache, file, indent=4)
        except IOError as e:
            logger.error(f"Error saving pricing cache: {e}")

        return model_total_cost, current_request_cost

    def invoke(self, *args, **kwargs):
        budget_config = getattr(self, "_budget_config", {})
        budget_mode = str(budget_config.get("budget_mode", BudgetMode.DISABLED)).capitalize()
        budget_limit = float(budget_config.get("budget_limit", 0.0))

        model_total_cost = self.get_current_model_cost()

        alert_message = ""
        if budget_limit > 0 and model_total_cost > budget_limit:
            alert_message = f"⚠️ **Budget Limit Exceeded!** Budget Limit: ${budget_limit:.6f}"
            logger.warning(alert_message)

        if budget_mode == BudgetMode.BLOCK.value and model_total_cost > budget_limit:
            return AIMessage(
                content="⛔ **Invocation Blocked Due to Budget Constraints.**\n"
                "Your request cannot be processed because the total cost has exceeded the budget limit.\n"
                "💰 **Cost Breakdown:**\n"
                f"   - 🎯 **Budget Limit:** `${budget_limit:.6f}`\n"
                f"   - 📊 Total Cost: `${model_total_cost:.6f}`"
            )

        response = super().invoke(*args, **kwargs)

        try:
            usage_metadata = response.usage_metadata
            model_total_cost, current_request_cost = self.compute_invocation_cost(**usage_metadata)

            response.usage_metadata["current_request_cost"] = round(current_request_cost, 6)
            response.usage_metadata["model_total_cost"] = round(model_total_cost, 6)

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


class NovaLLMConfig(LLMSettings):
    """Configuration for Amazon Nova LLMs (Pro and Lite)."""

    model_id: str = Field(
        default=DEFAULT_MODEL_ID,
        description=(
            "The Amazon Nova model ID. "
            "The region is determined by the AWS client configuration.\n"
            "Supported models:\n"
            f"Nova v1:\n"
            f"- Nova Pro: {NOVA_PRO_MODEL_ID}\n"
            f"- Nova Lite: {NOVA_LITE_MODEL_ID}\n"
            f"Nova 2 (next-gen):\n"
            f"- Nova 2 Lite: {NOVA_2_LITE_MODEL_ID}"
        ),
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Controls randomness in responses. Lower values (e.g., 0.2) make output more focused and deterministic, higher values (e.g., 0.8) make it more creative. Range: 0.0 to 1.0",
    )
    max_tokens: int = Field(
        default=4096,
        ge=1,
        le=100000,
        description="Maximum number of tokens to generate in the response.",
    )
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling: only consider tokens with cumulative probability up to this value. Range: 0.0 to 1.0",
    )
    model_kwargs: Optional[str] = Field(
        default="{}",
        description="Additional keyword arguments for the model in JSON string format.",
    )
    input_token_price: float = Field(
        default=NOVA_PRO_INPUT_PRICE,
        description=f"The price per {NOVA_TOKEN_UNIT} input tokens (in USD). Default is Nova Pro pricing.",
    )
    output_token_price: float = Field(
        default=NOVA_PRO_OUTPUT_PRICE,
        description=f"The price per {NOVA_TOKEN_UNIT} output tokens (in USD). Default is Nova Pro pricing.",
    )
    budget_mode: BudgetMode = Field(
        default=BudgetMode.DISABLED,
        description=(
            "The budget mode for the model, which controls cost monitoring and enforcement. "
            "Options:\n"
            "Disabled: No budget tracking or restrictions.\n"
            "Monitor: Logs the cost of each invocation without any notifications or enforcement.\n"
            "Notify: Sends a warning notification when the budget limit is exceeded.\n"
            "Trace: Appends cost breakdown details to the model's response.\n"
            "Block: Prevents further invocations once the budget limit is exceeded."
        ),
    )
    budget_limit: Optional[float] = Field(
        default=0.0,
        description="The maximum budget for the model in USD.",
    )
    _pyclass: Type = NovaLLM

    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "Amazon Nova",
            "description": "Amazon Nova LLMs (Pro and Lite) - Powerful foundation models from AWS Bedrock",
            "link": "https://aws.amazon.com/bedrock/",
        },
        arbitrary_types_allowed=True,
        use_enum_values=True,
        validate_assignment=True,
        extra="allow",
    )


class ClaudeLLM(ChatBedrockConverse):
    """Custom ChatBedrockConverse class for Anthropic Claude models via AWS Bedrock."""

    def __init__(self, **kwargs):
        model_id = kwargs.get("model_id", DEFAULT_CLAUDE_MODEL_ID)
        default_input_price, default_output_price = get_default_claude_pricing(model_id)

        model_kwargs = kwargs.get("model_kwargs", "{}")
        if isinstance(model_kwargs, str):
            try:
                model_kwargs = json.loads(model_kwargs)
            except json.JSONDecodeError:
                model_kwargs = {}

        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 4096)
        top_p = kwargs.get("top_p", 0.9)

        input_kwargs = {
            "model": model_id,
            "client": Boto3().get_client("bedrock-runtime"),
            "temperature": float(temperature) if temperature is not None else None,
            "max_tokens": int(max_tokens) if max_tokens is not None else None,
            "top_p": float(top_p) if top_p is not None else None,
        }

        if model_kwargs:
            input_kwargs["additional_model_request_fields"] = model_kwargs

        input_kwargs = {k: v for k, v in input_kwargs.items() if v is not None}

        super(ClaudeLLM, self).__init__(**input_kwargs)

        if kwargs.get("budget_mode", "Disabled") != "Disabled":
            budget_limit = kwargs.get("budget_limit", 0.0)
            input_price = kwargs.get("input_token_price", default_input_price)
            output_price = kwargs.get("output_token_price", default_output_price)

            def parse_float(value, default=0.0):
                if isinstance(value, (int, float)):
                    return float(value)
                return float(value) if str(value).replace(".", "", 1).isdigit() else default

            setattr(
                ClaudeLLM,
                "_budget_config",
                {
                    "budget_limit": parse_float(budget_limit),
                    "input_token_price": parse_float(input_price) / CLAUDE_TOKEN_UNIT,
                    "output_token_price": parse_float(output_price) / CLAUDE_TOKEN_UNIT,
                    "budget_mode": kwargs.get("budget_mode", "Disabled"),
                },
            )

    def get_current_model_cost(self):
        cached_cost_file = get_cached_cost_file_path()
        if os.path.exists(cached_cost_file):
            try:
                with open(cached_cost_file, "r") as f:
                    return float((json.load(f) or {}).get("current_cost", 0.0))
            except (json.JSONDecodeError, IOError):
                pass
        return 0.0

    def compute_invocation_cost(self, input_tokens, output_tokens, total_tokens):
        budget_config = getattr(self, "_budget_config", {})
        input_cost = budget_config.get("input_token_price", 0.0) * input_tokens
        output_cost = budget_config.get("output_token_price", 0.0) * output_tokens
        current_request_cost = round(input_cost + output_cost, 6)
        model_total_cost = self.get_current_model_cost() + current_request_cost
        try:
            with open(get_cached_cost_file_path(), "w") as f:
                json.dump({"current_cost": model_total_cost}, f, indent=4)
        except IOError as e:
            logger.error(f"Error saving pricing cache: {e}")
        return model_total_cost, current_request_cost

    def invoke(self, *args, **kwargs):
        budget_config = getattr(self, "_budget_config", {})
        budget_mode = str(budget_config.get("budget_mode", BudgetMode.DISABLED)).capitalize()
        budget_limit = float(budget_config.get("budget_limit", 0.0))
        model_total_cost = self.get_current_model_cost()

        alert_message = ""
        if budget_limit > 0 and model_total_cost > budget_limit:
            alert_message = f"⚠️ **Budget Limit Exceeded!** Budget Limit: ${budget_limit:.6f}"
            logger.warning(alert_message)

        if budget_mode == BudgetMode.BLOCK.value and model_total_cost > budget_limit:
            return AIMessage(
                content="⛔ **Invocation Blocked Due to Budget Constraints.**\n"
                "Your request cannot be processed because the total cost has exceeded the budget limit.\n"
                f"   - 🎯 **Budget Limit:** `${budget_limit:.6f}`\n"
                f"   - 📊 Total Cost: `${model_total_cost:.6f}`"
            )

        response = super().invoke(*args, **kwargs)

        try:
            usage_metadata = response.usage_metadata
            model_total_cost, current_request_cost = self.compute_invocation_cost(**usage_metadata)
            response.usage_metadata["current_request_cost"] = round(current_request_cost, 6)
            response.usage_metadata["model_total_cost"] = round(model_total_cost, 6)

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


class ClaudeLLMConfig(LLMSettings):
    """Configuration for Anthropic Claude models via AWS Bedrock."""

    model_id: str = Field(
        default=DEFAULT_CLAUDE_MODEL_ID,
        description=(
            "The Anthropic Claude model ID on AWS Bedrock.\n"
            "Supported models:\n"
            f"Claude Sonnet:\n"
            f"- Claude Sonnet 4.6 (latest): {CLAUDE_SONNET_4_6_MODEL_ID}\n"
            f"- Claude Sonnet 4.5: {CLAUDE_SONNET_4_5_MODEL_ID}\n"
            f"- Claude Sonnet 4: {CLAUDE_SONNET_4_MODEL_ID}\n"
            f"Claude Haiku:\n"
            f"- Claude Haiku 4.5: {CLAUDE_HAIKU_4_5_MODEL_ID}\n"
            f"Claude Opus:\n"
            f"- Claude Opus 4.6 (latest): {CLAUDE_OPUS_4_6_MODEL_ID}\n"
            f"- Claude Opus 4.5: {CLAUDE_OPUS_4_5_MODEL_ID}"
        ),
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Controls randomness in responses. Range: 0.0 to 1.0",
    )
    max_tokens: int = Field(
        default=4096,
        ge=1,
        le=200000,
        description="Maximum number of tokens to generate in the response.",
    )
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter. Range: 0.0 to 1.0",
    )
    model_kwargs: Optional[str] = Field(
        default="{}",
        description="Additional keyword arguments for the model in JSON string format.",
    )
    input_token_price: float = Field(
        default=CLAUDE_SONNET_4_INPUT_PRICE,
        description=f"The price per {CLAUDE_TOKEN_UNIT} input tokens (in USD). Default is Claude Sonnet 4 pricing.",
    )
    output_token_price: float = Field(
        default=CLAUDE_SONNET_4_OUTPUT_PRICE,
        description=f"The price per {CLAUDE_TOKEN_UNIT} output tokens (in USD). Default is Claude Sonnet 4 pricing.",
    )
    budget_mode: BudgetMode = Field(
        default=BudgetMode.DISABLED,
        description=(
            "The budget mode for the model.\n"
            "Disabled: No budget tracking.\n"
            "Monitor: Logs cost per invocation.\n"
            "Notify: Warns when budget exceeded.\n"
            "Trace: Appends cost to response.\n"
            "Block: Stops invocations over budget."
        ),
    )
    budget_limit: Optional[float] = Field(
        default=0.0,
        description="The maximum budget for the model in USD.",
    )
    _pyclass: Type = ClaudeLLM

    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "Anthropic Claude (Bedrock)",
            "description": "Anthropic Claude models (Sonnet, Haiku, Opus) via AWS Bedrock",
            "link": "https://aws.amazon.com/bedrock/claude/",
        },
        arbitrary_types_allowed=True,
        use_enum_values=True,
        validate_assignment=True,
        extra="allow",
    )


@hook
def agent_prompt_prefix(prefix, cat):
    # Append instruction to hide cost info, don't replace the entire prefix
    cost_instruction = "\nPlease do not include any cost breakdowns, request costs, or total cost information in responses."
    return prefix + cost_instruction


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
        cached_cost_file = get_cached_cost_file_path()
        with open(cached_cost_file, "w") as f:
            json.dump({}, f)
        return "✅ Cumulative model cost has been reset."
    except Exception as e:
        return f"❌ Error resetting cumulative model cost: {str(e)}"


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
        cached_cost_file = get_cached_cost_file_path()
        if not os.path.exists(cached_cost_file):
            return "⚠️ No cost data found. The cache might be empty."

        with open(cached_cost_file, "r") as f:
            cost_data = json.load(f)

        total_cost = cost_data.get("current_cost", 0.0)

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
        "How much does Nova Pro charge per token?",
        "What is the pricing for Nova Lite?",
        "Show the token price for Nova models.",
    ],
)
def get_current_model_pricing(data, cat):
    """Retrieve the pricing information for Amazon Nova models.

    Returns the cost per token for Nova Pro and Nova Lite models.
    """
    return (
        f"💲 **Amazon Nova Pricing**\n\n"
        f"**Nova Pro:**\n"
        f"🔹 **Input Cost:** ${NOVA_PRO_INPUT_PRICE:.6f} per {NOVA_TOKEN_UNIT} tokens\n"
        f"🔹 **Output Cost:** ${NOVA_PRO_OUTPUT_PRICE:.6f} per {NOVA_TOKEN_UNIT} tokens\n\n"
        f"**Nova Lite:**\n"
        f"🔹 **Input Cost:** ${NOVA_LITE_INPUT_PRICE:.6f} per {NOVA_TOKEN_UNIT} tokens\n"
        f"🔹 **Output Cost:** ${NOVA_LITE_OUTPUT_PRICE:.6f} per {NOVA_TOKEN_UNIT} tokens\n\n"
        f"**Nova 2 Lite (next-gen):**\n"
        f"🔹 **Input Cost:** ${NOVA_2_LITE_INPUT_PRICE:.6f} per {NOVA_TOKEN_UNIT} tokens\n"
        f"🔹 **Output Cost:** ${NOVA_2_LITE_OUTPUT_PRICE:.6f} per {NOVA_TOKEN_UNIT} tokens"
    )


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

    Returns the name and ID of the model in use.
    """
    return (
        f"🤖 **Current Model Information**\n"
        f"🔹 **Model:** Amazon Nova\n"
        f"🔹 **Available Models:**\n"
        f"   **Nova v1:**\n"
        f"   - Nova Pro: `{NOVA_PRO_MODEL_ID}`\n"
        f"   - Nova Lite: `{NOVA_LITE_MODEL_ID}`\n"
        f"   **Nova 2 (next-gen):**\n"
        f"   - Nova 2 Lite: `{NOVA_2_LITE_MODEL_ID}`\n"
        "Amazon Nova models are powerful foundation models from AWS Bedrock.\n"
        "Note: The region is determined by the AWS client configuration."
    )


@hook
def factory_allowed_llms(allowed, cat) -> List:
    return allowed + [NovaLLMConfig, ClaudeLLMConfig]


@plugin
def settings_model():
    """Plugin settings schema - this makes settings visible in the plugin page."""
    from pydantic import BaseModel

    class NovaPluginSettings(BaseModel):
        """Settings for Amazon Nova LLM plugin."""
        pass

    return NovaPluginSettings
