from __future__ import annotations

import ast
from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompt_values import ChatPromptValue

SOURCE_PATH = Path(__file__).with_name("bedrock_llms.py")
TARGET_FUNCTIONS = {"_normalize_bedrock_messages", "_normalize_bedrock_input"}


def _load_normalizers():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(SOURCE_PATH))
    selected_nodes = [
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name in TARGET_FUNCTIONS
    ]
    compiled = compile(ast.Module(body=selected_nodes, type_ignores=[]), str(SOURCE_PATH), "exec")
    namespace = {
        "AIMessage": AIMessage,
        "BaseMessage": BaseMessage,
        "HumanMessage": HumanMessage,
        "SystemMessage": SystemMessage,
        "ChatPromptValue": ChatPromptValue,
        "Any": object,
    }
    exec(compiled, namespace)
    return namespace["_normalize_bedrock_messages"], namespace["_normalize_bedrock_input"]


def main() -> int:
    normalize_messages, normalize_input = _load_normalizers()

    system = SystemMessage(content="system")
    user = HumanMessage(content="ciao")
    ai_1 = AIMessage(content="prima risposta")
    ai_2 = AIMessage(content="seconda risposta")

    normalized, dropped = normalize_messages([system, ai_1, ai_2, user])
    assert dropped == 2, f"attesi 2 messaggi assistant rimossi, ottenuto {dropped}"
    assert isinstance(normalized[0], SystemMessage)
    assert isinstance(normalized[1], HumanMessage)

    normalized_prompt, dropped_prompt = normalize_input(ChatPromptValue(messages=[system, ai_1, user]))
    assert dropped_prompt == 1, f"atteso 1 assistant rimosso dal ChatPromptValue, ottenuto {dropped_prompt}"
    assert isinstance(normalized_prompt, ChatPromptValue)
    assert isinstance(normalized_prompt.messages[0], SystemMessage)
    assert isinstance(normalized_prompt.messages[1], HumanMessage)

    already_valid, dropped_valid = normalize_messages([system, user, ai_1])
    assert dropped_valid == 0
    assert len(already_valid) == 3
    assert isinstance(already_valid[1], HumanMessage)

    no_user, dropped_no_user = normalize_messages([system, ai_1])
    assert dropped_no_user == 0, "non deve alterare una sequenza senza user se non può ripararla in sicurezza"
    assert len(no_user) == 2
    assert isinstance(no_user[1], AIMessage)

    print("Bedrock message order normalization: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

