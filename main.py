import os
from enum import Enum
from typing import Any

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

from chat_models import ChatModel, ChatOllama, ChatOpenAI

load_dotenv()


def main():
    information = """
        Elon Reeve Musk FRS (/ˈiːlɒn/ EE-lon; born June 28, 1971) is a businessman, known for his leadership of Tesla, SpaceX, X (formerly Twitter), and the Department of Government Efficiency (DOGE). Musk has been the wealthiest person in the world since 2021; as of May 2025, Forbes estimates his net worth to be US$424.7 billion.

        Born to a wealthy family in Pretoria, South Africa, Musk emigrated in 1989 to Canada. He received bachelor's degrees from the University of Pennsylvania in 1997 before moving to California, United States, to pursue business ventures. In 1995, Musk co-founded the software company Zip2. Following its sale in 1999, he co-founded X.com, an online payment company that later merged to form PayPal, which was acquired by eBay in 2002. That year, Musk also became an American citizen.

        In 2002, Musk founded the space technology company SpaceX, becoming its CEO and chief engineer; the company has since led innovations in reusable rockets and commercial spaceflight. Musk joined the automaker Tesla as an early investor in 2004 and became its CEO and product architect in 2008; it has since become a leader in electric vehicles. In 2015, he co-founded OpenAI to advance artificial intelligence (AI) research but later left; growing discontent with the organization's direction and their leadership in the AI boom in the 2020s led him to establish xAI. In 2022, he acquired the social network Twitter, implementing significant changes and rebranding it as X in 2023. His other businesses include the neurotechnology company Neuralink, which he co-founded in 2016, and the tunneling company the Boring Company, which he founded in 2017.

        Musk was the largest donor in the 2024 U.S. presidential election, and is a supporter of global far-right figures, causes, and political parties. In early 2025, he served as senior advisor to United States president Donald Trump and as the de facto head of DOGE. After a public feud with Trump, Musk left the Trump administration and announced he was creating his own political party, the America Party.

        Musk's political activities, views, and statements have made him a polarizing figure, especially following the COVID-19 pandemic. He has been criticized for making unscientific and misleading statements, including COVID-19 misinformation and promoting conspiracy theories, and affirming antisemitic, racist, and transphobic comments. His acquisition of Twitter was controversial due to a subsequent increase in hate speech and the spread of misinformation on the service. His role in the second Trump administration attracted public backlash, particularly in response to DOGE.
    """
    summary_template = """
    given the information {information} about Elon Musk I want you to create:
    1. A short summary
    2. Two interesting facts about him
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    # llm = ChatFactory.create(Provider.OPENAI, model="gpt-5-nano", temperature=0)
    llm = ChatFactory.create(
        Provider.OLLAMA,
        model="qwen3:1.7b",
        temperature=0,
        base_url="http://192.168.88.30:11434",
    )
    prompt_text = summary_prompt_template.format(information=information)
    response_text = llm.chat(prompt_text)
    print(response_text)

    # the code can be traced at
    # https://smith.langchain.com/o/86c543d4-8b8d-40dd-8506-94c9c9fa697d/projects/p/31e67965-f3f0-4d0a-ba73-58e2bf042144?timeModel=%7B%22duration%22%3A%221d%22%7D


class Provider(Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"


class ChatFactory:
    """Factory for creating chat model instances by provider."""

    @staticmethod
    def create(provider: Provider, **config: Any) -> ChatModel:
        match provider:
            case Provider.OPENAI:
                return ChatOpenAI(
                    model=config.get("model", "gpt-5-nano"),
                    temperature=config.get("temperature", 0),
                )
            case Provider.OLLAMA:
                return ChatOllama(
                    model=config.get("model", "gemma3:4b"),
                    temperature=config.get("temperature", 0),
                    base_url=config.get("base_url"),
                )
            case _:
                raise ValueError(f"Unknown provider type: {provider}")


if __name__ == "__main__":
    main()
