import os
from crewai import Crew, Process
from dotenv import load_dotenv
from src.agents import build_agents
from src.tasks import build_tasks

def run_crew(start: str, end: str, use_llm: bool = True):
    load_dotenv()
    agents = build_agents(use_llm=use_llm)
    tasks = build_tasks(agents, start, end)
    crew = Crew(
        agents=list(agents.values()),
        tasks=tasks,
        process=Process.sequential
    )
    # kickoff returns a dict-like structure with task outputs in newer versions;
    # for compatibility, just print the final output.
    result = crew.kickoff()
    return result
