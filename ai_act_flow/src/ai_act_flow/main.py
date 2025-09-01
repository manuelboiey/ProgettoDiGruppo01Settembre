#!/usr/bin/env python
from random import randint

from pydantic import BaseModel

from crewai.flow import Flow, listen, start

from ai_act_flow.crews.rag_crew.rag_crew import RagCrew
from questions import questions


class FlowState(BaseModel):
    sentence_count: int = 1
    poem: str = ""


class AiActFlow(Flow[FlowState]):

   @start()
   def start(self):
    crew = RagCrew()
    answers = []
    for question in questions:
       answers.append(crew.kickoff(inputs=question))

    

    self.state = FlowState()

def kickoff():
    poem_flow = AiActFlow()
    poem_flow.kickoff()


def plot():
    poem_flow = AiActFlow()
    poem_flow.plot()


if __name__ == "__main__":
    kickoff()
