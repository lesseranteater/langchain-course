from __future__ import annotations

import json
import os
from typing import Any, Callable
from urllib import error, request

from dotenv import load_dotenv

load_dotenv()

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://192.168.88.30:11434")


def get_product_price(product_name: str) -> float:
    print(f"Tool called: get_product_price with argument '{product_name}'")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product_name.lower(), 0)


def apply_discount(price: float, discount_tier: str) -> float:
    print(
        f"Tool called: apply_discount with arguments price={price}, discount_tier={discount_tier}"
    )
    discount_mapping = {"bronze": 5, "silver": 10, "gold": 25}
    discount_percentage = discount_mapping.get(discount_tier.lower(), 0)
    return round(price * (1 - discount_percentage / 100), 2)


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Get the price of a product.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {
                        "type": "string",
                        "description": "The product name to look up.",
                    }
                },
                "required": ["product_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a customer discount tier to a product price.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {
                        "type": "number",
                        "description": "The product price returned by get_product_price.",
                    },
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier to apply, such as bronze, silver, or gold.",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]

TOOL_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount,
}

SYSTEM_PROMPT = (
    "You are a helpful shopping assistant. "
    "You have access to a product catalog tool "
    "and a discount tool.\n\n"
    "STRICT RULES — you must follow these exactly:\n"
    "1. NEVER guess or assume any product price. "
    "You MUST call get_product_price first to get the real price.\n"
    "2. Only call apply_discount AFTER you have received "
    "a price from get_product_price. Pass the exact price "
    "returned by get_product_price — do NOT pass a made-up number.\n"
    "3. NEVER calculate discounts yourself using math. "
    "Always use the apply_discount tool.\n"
    "4. If the user does not specify a discount tier, "
    "ask them which tier to use — do NOT assume one."
)


def post_ollama_chat(
    messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools

    body = json.dumps(payload).encode("utf-8")
    api_url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat"
    http_request = request.Request(
        api_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(http_request) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Ollama chat request failed with HTTP {exc.code}: {error_body}"
        ) from exc
    except error.URLError as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {api_url}: {exc.reason}"
        ) from exc


def normalize_tool_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments

    if isinstance(arguments, str):
        parsed_arguments = json.loads(arguments)
        if isinstance(parsed_arguments, dict):
            return parsed_arguments

    raise ValueError(f"Unsupported tool arguments payload: {arguments!r}")


def build_assistant_message(message: dict[str, Any]) -> dict[str, Any]:
    assistant_message = {"role": "assistant"}

    for key in ("content", "thinking", "tool_calls"):
        value = message.get(key)
        if value:
            assistant_message[key] = value

    return assistant_message


def run_agent(question: str) -> str | None:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    print(f"Initial question: {question}")
    print("=" * 60)

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        response = post_ollama_chat(messages, tools=TOOLS)
        assistant_message = response.get("message", {})
        tool_calls = assistant_message.get("tool_calls") or []
        assistant_content = assistant_message.get("content", "")

        messages.append(build_assistant_message(assistant_message))

        if assistant_content:
            print(f"  [Assistant] {assistant_content}")

        if not tool_calls:
            print(f"\nFinal Answer: {assistant_content}")
            return assistant_content

        for tool_call in tool_calls:
            function_payload = tool_call.get("function", {})
            tool_name = function_payload.get("name")
            tool_arguments = normalize_tool_arguments(
                function_payload.get("arguments", {})
            )

            print(f"  [Tool Selected] {tool_name} with args: {tool_arguments}")

            tool_to_use = TOOL_FUNCTIONS.get(tool_name)
            if tool_to_use is None:
                raise ValueError(f"Tool '{tool_name}' not found")

            observation = tool_to_use(**tool_arguments)

            print(f"  [Tool Result] {observation}")

            messages.append(
                {
                    "role": "tool",
                    "tool_name": tool_name,
                    "content": str(observation),
                }
            )

    print("ERROR: Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Hello Ollama JSON Agent!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")
    print("\nAgent loop finished.")
    print(f"Result: {result}")
