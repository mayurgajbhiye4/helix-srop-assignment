"""
SROP Root Orchestrator — Google ADK agent.

Routes every user turn to KnowledgeAgent or AccountAgent via ADK's AgentTool.
This means the LLM decides which tool to call — you do not parse its output.

Intent → sub-agent:
  knowledge:  "how do I X", "what is X", docs questions
  account:    "show my builds", "my account status", usage questions
  escalation:   "create a ticket", "I need support", urgent issues
  smalltalk:  greetings, thanks — root agent handles inline (no tool call)

See docs/google-adk-guide.md for AgentTool pattern and event extra
"""

from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool

from app.agents.tools.account_tools import get_account_status, get_recent_builds
from app.agents.tools.escalation_tools import create_ticket
from app.agents.tools.search_docs import search_docs
from app.settings import settings

ROOT_INSTRUCTION = """
You are the Helix Support Concierge — a routing agent.
Call the correct specialist tool based on the user's intent.

Intent → tool:
- HOW to do something, WHAT something is, docs/feature questions → knowledge_agent
- Their account, builds, status, usage → account_agent
- Create a support ticket, escalate an issue, urgent/critical problems → escalation_agent
- Greetings or off-topic → respond directly, no tool call

❌ Out-of-scope topics (REFUSE these):
- Creative writing (poems, stories, fiction)
- General knowledge unrelated to Helix
- Personal advice or medical/legal guidance
- Coding tasks outside Helix context

When user asks out-of-scope: Say "I'm specialized in Helix support.
This question is outside my scope. How can I help with your Helix needs?"

Always call a tool when intent matches. Never answer knowledge or account questions yourself.
User context will be in the system message — use it.
When escalating, always include a clear summary and set priority appropriately.
"""

KNOWLEDGE_INSTRUCTION = """
You are KnowledgeAgent for Helix product documentation.
Answer product, troubleshooting, setup, and "how do I" questions by calling search_docs.
Use the returned chunks as your source of truth and cite chunk IDs in the answer.
If the docs do not contain enough evidence, say what is missing.
"""

ACCOUNT_INSTRUCTION = """
You are AccountAgent for Helix user account and build questions.
Use the user_id from the provided context when calling account tools.
Call get_recent_builds for build history questions.
Call get_account_status for plan, usage, limits, and account status questions.
Be concise and include concrete IDs or limits returned by the tools.
"""

ESCALATION_INSTRUCTION = """
You are EscalationAgent for Helix support ticket creation.
When a user needs to escalate an issue or create a support ticket:
1. Extract a clear summary of the problem from the user's message
2. Determine the priority: "low", "medium", "high", or "critical"
3. Call create_ticket with user_id, summary, and priority
4. Return the ticket_id and confirm the ticket was created
Always confirm the ticket creation with the ticket_id in your response.
"""

knowledge_agent = LlmAgent(
    name="knowledge_agent",
    model=settings.adk_model,
    instruction=KNOWLEDGE_INSTRUCTION,
    tools=[search_docs],
)

account_agent = LlmAgent(
    name="account_agent",
    model=settings.adk_model,
    instruction=ACCOUNT_INSTRUCTION,
    tools=[get_recent_builds, get_account_status],
)

escalation_agent = LlmAgent(
    name="escalation_agent",
    model=settings.adk_model,
    instruction=ESCALATION_INSTRUCTION,
    tools=[create_ticket],
)

knowledge_tool = AgentTool(agent=knowledge_agent)
account_tool = AgentTool(agent=account_agent)
escalation_tool = AgentTool(agent=escalation_agent)

root_agent = LlmAgent(
    name="srop_root",
    model=settings.adk_model,
    instruction=ROOT_INSTRUCTION,
    tools=[knowledge_tool, account_tool, escalation_tool],
)
