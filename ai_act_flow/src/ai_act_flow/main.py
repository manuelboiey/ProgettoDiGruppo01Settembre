#!/usr/bin/env python
from random import randint

from pydantic import BaseModel

from crewai.flow import Flow, listen, start

from ai_act_flow.crews.rag_crew.rag_crew import RagCrew
from ai_act_flow.crews.docs_generation_crew.docs_generation_crew import DocsGenerationCrew
from questions import questions


class FlowState(BaseModel):
    questions: list[str]
    answers: list[str]


class AiActFlow(Flow[FlowState]):

    @start()
    def ask_rag(self):
        self.state.questions = questions
        crew = RagCrew()
        for question in self.state.questions:
            self.state.answers.append(crew.kickoff(inputs=question))
        
        return self.state
    
    @listen(ask_rag)
    def ask_user(self):
        for i, answer in self.state.answers.enumerate(self.state.answers):
            if answer == 'Not in documentation':        
                print(f"{self.state.questions[i]}")
                self.state.answers[i] = input()
        return self.state
    
    @listen(ask_user)
    def build_ai_documentation(self):
        crew = DocsGenerationCrew()

        with open('templates/model_documentation.md', 'r') as f:
            ai_act_documentation = f.read()

        # Build input
        input = 'QUESTIONS AND ANSWERS:\n'
        input.append('\n'.join([f"{i+1}. Question: {q} Answer: {self.state.answers[i]}" for i, q in enumerate(self.state.questions)]))
        input.append('\n\nTEMPLATE:\n')
        input.append(ai_act_documentation)

        document = crew.kickoff(inputs=input)
        with open("EU_AI_Act_Documentation.md", "w") as f:
            f.write(document)
        return self.state


def kickoff():
    poem_flow = AiActFlow()
    poem_flow.kickoff()


def plot():
    poem_flow = AiActFlow()
    poem_flow.plot()


if __name__ == "__main__":
    kickoff()
