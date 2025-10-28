from agents.base_agent import BaseAgent
from prompts.calendar_agent_prompt import CALENDAR_AGENT_PROMPT
from prompts.email_agent_prompt import EMAIL_AGENT_PROMPT
from tools.all_tools import create_calendar_event, get_available_time_slots, send_email
from langchain.agents.middleware import HumanInTheLoopMiddleware

# Calendar Agent
calendar_agent = BaseAgent(
    tools=[create_calendar_event, get_available_time_slots],
    system_prompt=CALENDAR_AGENT_PROMPT,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"create_calendar_event": True},
            description_prefix="Calendar event pending approval",
        ),
    ],
).create()

# Email Agent
email_agent = BaseAgent(tools=[send_email],system_prompt=EMAIL_AGENT_PROMPT).create()