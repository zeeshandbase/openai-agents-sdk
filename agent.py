"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  agent.py — Complete OpenAI Agents SDK Reference (All Concepts)              ║
║                                                                              ║
║  Industry Problem: AI Customer Support Agent for a SaaS Company              ║
║  - Classifies tickets (structured output)                                    ║
║  - Looks up customer data (function tools)                                   ║
║  - Checks system status (async tools)                                        ║
║  - Generates responses (agent-as-tool)                                       ║
║  - Runs locally via Ollama (zero cost)                                       ║
║                                                                              ║
║  Run: uv run agent.py                                                        ║
║  Requires: uv add openai-agents openai ollama pydantic                       ║
║  Requires: Ollama running with llama3.1:8b (ollama pull llama3.1:8b)         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from agents import (
    Agent,
    Runner,
    RunResult,
    OpenAIChatCompletionsModel,
    ModelSettings,
    function_tool,
)
from agents.tracing import set_tracing_disabled


# ============================================================================
# CONCEPT 1: LOCAL MODEL SETUP (Ollama)
# ============================================================================
# Why: Run agents locally for free, with full data privacy.
# How: Point AsyncOpenAI client to Ollama's OpenAI-compatible endpoint.
# Key: Disable tracing (it sends data to OpenAI by default).

set_tracing_disabled(True)  # No telemetry to OpenAI

ollama_client = AsyncOpenAI(
    base_url="http://localhost:11434/v1",    # Ollama's endpoint
    api_key="rizwan",                        # Dummy key (required by client)
)

local_model = OpenAIChatCompletionsModel(
    model="llama3.1:8b",                     # Must match `ollama list`
    openai_client=ollama_client,
)


# ============================================================================
# CONCEPT 2: STRUCTURED OUTPUT (Pydantic Models)
# ============================================================================
# Why: Instead of parsing messy text, get clean typed data from the LLM.
# How: Define a Pydantic model and set Agent(output_type=MyModel).
# Key: The LLM is forced to return JSON matching your schema.

class TicketClassification(BaseModel):
    """Structured output for ticket classification."""
    category: Literal["billing", "technical", "account", "general"] = Field(
        description="Which department handles this ticket"
    )
    priority: Literal["P1-critical", "P2-high", "P3-medium", "P4-low"] = Field(
        description="Ticket priority level"
    )
    sentiment: Literal["angry", "frustrated", "neutral", "positive"] = Field(
        description="Customer's emotional state"
    )
    summary: str = Field(
        description="One-line summary for the support dashboard"
    )


# ============================================================================
# CONCEPT 3: FUNCTION TOOLS (@function_tool)
# ============================================================================
# Why: Tools let agents take ACTIONS, not just generate text.
# How: Decorate any Python function with @function_tool.
# Key: Docstring becomes the tool description (LLM reads it to decide when to call).
#      Type hints become the JSON schema (LLM uses them to format arguments).

@function_tool
def lookup_customer(email: str) -> str:
    """Look up customer details by their email address."""
    # In production: query your CRM/database
    customers = {
        "ahmed@example.com": {
            "name": "Ahmed Hassan",
            "plan": "Enterprise",
            "since": "2023-01",
            "mrr": "$299/mo",
            "tickets_open": 2,
        },
        "sara@startup.io": {
            "name": "Sara Khan",
            "plan": "Pro",
            "since": "2024-06",
            "mrr": "$49/mo",
            "tickets_open": 0,
        },
    }
    customer = customers.get(email.lower())
    if not customer:
        return f"No customer found with email: {email}"
    return (
        f"Customer: {customer['name']}\n"
        f"Plan: {customer['plan']} ({customer['mrr']})\n"
        f"Customer since: {customer['since']}\n"
        f"Open tickets: {customer['tickets_open']}"
    )


@function_tool
def check_service_status(service: str) -> str:
    """Check the current status of a company service/feature."""
    statuses = {
        "api":       "Operational (99.98% uptime, 45ms avg latency)",
        "dashboard": "Degraded (slow loading, team investigating)",
        "billing":   "Operational",
        "auth":      "Operational",
    }
    status = statuses.get(service.lower())
    if not status:
        return f"Unknown service: {service}. Available: {list(statuses.keys())}"
    return f"{service}: {status}"


@function_tool
def create_ticket(
    customer_email: str,
    category: str,
    priority: str,
    description: str,
) -> str:
    """Create a support ticket in the system."""
    ticket_id = f"TKT-{abs(hash(description)) % 100000:05d}"
    return (
        f"Ticket created!\n"
        f"ID: {ticket_id}\n"
        f"Customer: {customer_email}\n"
        f"Category: {category} | Priority: {priority}\n"
        f"Description: {description[:100]}...\n"
        f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )


@function_tool
def search_knowledge_base(query: str) -> str:
    """Search the internal knowledge base for help articles."""
    articles = {
        "password": "Article #101: Reset password at settings > security > change password. "
                    "If locked out, use 'Forgot Password' on login page.",
        "billing":  "Article #201: Invoices are generated on the 1st of each month. "
                    "Change plan at settings > billing > change plan.",
        "api":      "Article #301: API docs at docs.example.com/api. "
                    "Rate limit: 1000 req/min (Pro), 10000 req/min (Enterprise).",
        "export":   "Article #401: Export data at settings > data > export. "
                    "Supports CSV, JSON, and PDF formats.",
    }
    for key, article in articles.items():
        if key in query.lower():
            return article
    return "No relevant articles found. Escalate to human agent."


# ============================================================================
# CONCEPT 4: DYNAMIC INSTRUCTIONS (Function-based)
# ============================================================================
# Why: Inject runtime context (time, user data, system state) into instructions.
# How: Pass a function instead of a string to Agent(instructions=...).

def support_instructions(context, agent):
    """Dynamic instructions that change based on current state."""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""
    You are an AI customer support agent for "CloudSync" — a SaaS platform.
    Current time: {current_time}

    YOUR WORKFLOW (follow this order):
    1. Greet the customer professionally.
    2. Look up their account using lookup_customer if they provide an email.
    3. Understand their issue — search_knowledge_base for known solutions.
    4. If it's a service issue, check_service_status to see if it's a known outage.
    5. If you can solve it, provide the solution clearly.
    6. If you can't solve it, create_ticket to escalate.
    7. Always end with: "Is there anything else I can help with?"

    YOUR RULES:
    - Be empathetic, professional, and concise.
    - If the customer is angry, acknowledge their frustration first.
    - ALWAYS use tools to get real data — never make up information.
    - For billing issues on Enterprise plans, always escalate (create a ticket).
    - Include ticket ID when creating tickets.
    """


# ============================================================================
# CONCEPT 5: SPECIALIST AGENT + AGENT-AS-TOOL
# ============================================================================
# Why: Complex tasks benefit from specialist agents working together.
# How: Create a specialist agent, then use agent.as_tool() to let the
#      orchestrator call it like a regular tool.

classifier_agent = Agent(
    name="Ticket Classifier",
    instructions="""
    You classify customer support messages.
    Analyze the message and return structured classification data.
    Be accurate — wrong classification wastes everyone's time.
    
    Priority guide:
    - P1-critical: Service down, data loss, security issue
    - P2-high: Feature broken, payment failure, angry customer
    - P3-medium: Bug report, how-to question, feature request
    - P4-low: General feedback, suggestions
    """,
    model=local_model,
    output_type=TicketClassification,            # Forces structured JSON output
    model_settings=ModelSettings(temperature=0.1),  # Low temp = consistent results
)


# ============================================================================
# CONCEPT 6: THE MAIN AGENT (Orchestrator)
# ============================================================================
# This is the primary agent that the user interacts with.
# It has:
# - Dynamic instructions (function-based)
# - Multiple function tools
# - A specialist agent-as-tool
# - A specific model (local Ollama)
# - Custom model settings

support_agent = Agent(
    name="CloudSync Support",
    instructions=support_instructions,     # Dynamic (function, not string)
    model=local_model,                     # Runs locally via Ollama
    model_settings=ModelSettings(
        temperature=0.3,                   # Slightly creative but mostly focused
        max_tokens=1000,                   # Cap response length
    ),
    tools=[
        # Function tools (direct)
        lookup_customer,
        check_service_status,
        create_ticket,
        search_knowledge_base,
        # Agent-as-tool (nested specialist)
        classifier_agent.as_tool(
            tool_name="classify_ticket",
            tool_description="Classify a customer message into category, priority, and sentiment",
        ),
    ],
)


# ============================================================================
# CONCEPT 7: RUNNING THE AGENT
# ============================================================================
# Three ways to run:
#   Runner.run()          — async, returns RunResult
#   Runner.run_sync()     — sync wrapper (not in Jupyter/async contexts)
#   Runner.run_streamed() — async streaming

async def handle_customer(message: str) -> None:
    """Process a single customer message."""
    print(f"\n{'='*70}")
    print(f"Customer: {message}")
    print(f"{'='*70}")

    result: RunResult = await Runner.run(support_agent, message)

    print(f"\nAgent Response:\n{result.final_output}")
    print(f"\nAgent: {result.last_agent.name}")
    print(f"Items generated: {len(result.new_items)}")


# ============================================================================
# CONCEPT 8: MULTI-TURN CONVERSATION
# ============================================================================
# Why: Real support conversations span multiple messages.
# How: Use result.to_input_list() to carry history forward.

async def interactive_session():
    """Run an interactive multi-turn support session."""
    print("\n" + "=" * 70)
    print("CloudSync AI Support — Interactive Session")
    print("   Type 'quit' to exit")
    print("=" * 70)

    history = None  # Will hold conversation state

    while True:
        user_input = input("\n👤 You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("Thank you for contacting CloudSync Support!")
            break
        if not user_input:
            continue

        # Build input: either fresh or with conversation history
        if history is not None:
            agent_input = history + [{"role": "user", "content": user_input}]
        else:
            agent_input = user_input

        result = await Runner.run(support_agent, agent_input)

        # Save conversation history for next turn
        history = result.to_input_list()

        print(f"\nSupport: {result.final_output}")


# ============================================================================
# MAIN — Run demo scenarios
# ============================================================================

async def main():
    """Demonstrate the agent with realistic support scenarios."""

    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║           CloudSync AI Support Agent — Demo                  ║
    ║                                                              ║
    ║  Concepts demonstrated:                                      ║
    ║  - Local model (Ollama)        - Function tools              ║
    ║  - Structured output           - Agent-as-tool               ║
    ║  - Dynamic instructions        - Multi-turn conversations    ║
    ║  - Model settings              - Agentic workflow loop       ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    # ── Scenario 1: Technical issue with account lookup ──
    await handle_customer(
        "Hi, I'm ahmed@example.com. The dashboard has been super slow today. "
        "Is something wrong with the system?"
    )

    # ── Scenario 2: Billing question ──
    await handle_customer(
        "I need to know how to change my billing plan. Where do I find that?"
    )

    # ── Scenario 3: Angry customer with critical issue ──
    await handle_customer(
        "THIS IS UNACCEPTABLE! I'm sara@startup.io and your API has been "
        "returning errors for 2 hours. My production app is DOWN. Fix this NOW "
        "or I'm switching to a competitor!"
    )

    # ── Uncomment for interactive mode ──
    # await interactive_session()


if __name__ == "__main__":
    asyncio.run(main())
