import sys
from pathlib import Path

# 把项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # 根据你的文件位置调整
sys.path.append(str(PROJECT_ROOT))


from langchain_core.tools import tool
from agents.agent import calendar_agent, email_agent
from agents.base_agent import BaseAgent

@tool
def schedule_event(request: str) -> str:
    """Schedule calendar events using natural language.

    Use this when the user wants to create, modify, or check calendar appointments.
    Handles date/time parsing, availability checking, and event creation.

    Input: Natural language scheduling request (e.g., 'meeting with design team
    next Tuesday at 2pm')
    """
    result = calendar_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].text


@tool
def manage_email(request: str) -> str:
    """Send emails using natural language.

    Use this when the user wants to send notifications, reminders, or any email
    communication. Handles recipient extraction, subject generation, and email
    composition.

    Input: Natural language email request (e.g., 'send them a reminder about
    the meeting')
    """
    result = email_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].text

SUPERVISOR_PROMPT = (
    "You are a helpful personal assistant. "
    "You can schedule calendar events and send emails. "
    "Break down user requests into appropriate tool calls and coordinate the results. "
    "When a request involves multiple actions, use multiple tools in sequence."
)



supervisor_agent = BaseAgent(
   tools=[schedule_event, manage_email],
    system_prompt=SUPERVISOR_PROMPT,
).create()


query = "Schedule a team standup for tomorrow at 9am"

for step in supervisor_agent.stream(
    {"messages": [{"role": "user", "content": query}]}
):
    for update in step.values():
        print(update,type(update))
        for message in update.get("messages", []):
            message.pretty_print()