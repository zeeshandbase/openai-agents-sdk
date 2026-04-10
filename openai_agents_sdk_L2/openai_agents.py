import asyncio
from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel, function_tool

from agents.tracing import set_tracing_disabled


set_tracing_disabled(True)

@function_tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

local_model = OpenAIChatCompletionsModel(
    model="minimax-m2.7:cloud",
    openai_client=AsyncOpenAI(
        api_key="ollama",
        base_url="http://localhost:11434/v1"
    )
)

agent = Agent(
    name="History Tutor",
    instructions="You answers history questions clearly and concisely",
    model=local_model,
    tools=[get_weather]
)

async def main():
    # query = "Who built badshai masjid in lahore pakistan"
    # query = "what is the weather in lahore pakistan"
    query = "Hi"
    result = await Runner.run(agent, query)
    print(result.final_output)

asyncio.run(main())