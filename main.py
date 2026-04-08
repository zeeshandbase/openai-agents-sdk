# import asyncio
from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel, function_tool

from agents.tracing import set_tracing_disabled


set_tracing_disabled(True)

# @function_tool
# def get_weather(city: str) -> str:
#     """Get weather for a given city."""
#     return f"It's always sunny in {city}!"

local_model = OpenAIChatCompletionsModel(
    model="minimax-m2.7:cloud",
    openai_client=AsyncOpenAI(
        api_key="ollama",
        base_url="http://localhost:11434/v1"
    )
)

agent = Agent(
    name="History Tutor",
    instructions="You answer history questions clearly and concisely.",
    model=local_model,
    # tools=[get_weather]
)


# query = "Who built the Great Wall of China?"
# query = "What is the weather in lahore pakistan"
query = "Hi"
result = Runner.run_sync(agent, query)
print(result.final_output)
