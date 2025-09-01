from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from ai_act_flow.tools.rag import RAG


@CrewBase
class DocsGenerationCrew:
    """DocsGenerationCrew Crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def docs_writer_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["docs_writer_agent"]
        )

    @task
    def write_document(self) -> Task:
        return Task(
            config=self.tasks_config["write_document"],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the DocsGenerationCrew"""

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
