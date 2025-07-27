import asyncio
import logging

import fastapi
from autogen_agentchat.agents import UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.agents.web_surfer import MultimodalWebSurfer


app = fastapi.FastAPI(title="Web Scrapping Agent")
_log = logging.getLogger(__name__)

async def main() -> None:
    """Main function to run the web surfer agent with a user proxy agent."""
    model_client = OpenAIChatCompletionClient(model="gpt-4o")
    web_surfer = MultimodalWebSurfer("web_surfer", model_client, headless=True, animate_actions=False,
                                     start_page="https://google.com")
    user_proxy = UserProxyAgent("user_proxy")
    termination = TextMentionTermination("exit", sources=["user_proxy"])
    team = RoundRobinGroupChat([web_surfer, user_proxy], termination_condition=termination)
    try:
        await Console(team.run_stream(
            task="Find information about AutoGen and write a short summary, don't use Bing, use Google instead."))
    finally:
        await web_surfer.close()
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
