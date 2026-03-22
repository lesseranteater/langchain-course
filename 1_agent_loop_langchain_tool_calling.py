from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"


# --- Tools (LangChain @tool decorator) ---
@tool
def get_product_price(product_name: str) -> float:
    """Get the price of a product."""
    print(f"Tool called: get_product_price with argument '{product_name}'")
    # In a real implementation, this would query a database or an API.
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product_name.lower(), 0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount to a price."""
    print(
        f"Tool called: apply_discount with arguments price={price}, discount_tier={discount_tier}"
    )
    discount_mapping = {"bronze": 5, "silver": 10, "gold": 25}
    discount_percentage = discount_mapping.get(discount_tier.lower(), 0)
    return round(price * (1 - discount_percentage / 100), 2)


# --- Agent Loop ---
@traceable(name="LangChain Agent Loop")
def run_agent_loop(question: str):
    pass


if __name__ == "__main__":
    print("Hello LangChain Agent Loop!")
    run_agent_loop(
        question="What is the price of a laptop after applying a gold discount?"
    )
