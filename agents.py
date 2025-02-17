import os
from crewai import Agent
from crewai import LLM
from textwrap import dedent
from langchain_openai import ChatOpenAI
from LLMSelector2 import LLMHelper, LLM
from Tools.search_tools import SearchTools
from Tools.calculator_tools import CalculatorTools


"""
Creating Agents Cheat Sheet:
- Think like a boss. Work backwards from the goal and think which employee 
    you need to hire to get the job done.
- Define the Captain of the crew who orient the other agents towards the goal. 
- Define which experts the captain needs to communicate with and delegate tasks to.
    Build a top down structure of the crew.

Goal:
- Create a 7-day travel itinerary with detailed per-day plans,
    including budget, packing suggestions, and safety tips.

Captain/Manager/Boss:
- Expert Travel Agent

Employees/Experts to hire:
- City Selection Expert 
- Local Tour Guide


Notes:
- Agents should be results driven and have a clear goal in mind
- Role is their job title
- Goals should actionable
- Backstory should be their resume
"""

# groq_helper = LLMHelper(provider="groq")
# llm = groq_helper.get_llm()
# api_key =os.environ.get("GROQ_API_KEY")
# llm=LLM(model="groq/deepseek-r1-distill-qwen-32b", api_key=api_key, temperature=0.7, verify=False)
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")

llm = "gemini/gemini-2.0-flash"

class TravelAgents:

    def expert_travel_agent(self):
        return Agent(
            role="Expert Travel Agent",
            backstory=dedent(
                f"""Expert in travel planning and logistics. 
                I have decades of experience making travel itineraries."""),
            goal=dedent(f"""
                        Create a 7-day travel itinerary with detailed per-day plans,
                        include budget, packing suggestions, and safety tips.
                        """),
            tools=[
                SearchTools.search_internet,
                CalculatorTools.calculate
            ],
            verbose=True,
            llm=llm,
        )

    def city_selection_expert(self):
        return Agent(
            role="City Selection Expert",
            backstory=dedent(
                f"""Expert at analyzing travel data to pick ideal destinations"""),
            goal=dedent(
                f"""Select the best cities based on weather, season, prices, and traveler interests"""),
            tools=[SearchTools.search_internet],
            verbose=True,
            llm=llm,
        )

    def local_tour_guide(self):
        return Agent(
            role="Local Tour Guide",
            backstory=dedent(f"""Knowledgeable local guide with extensive information
        about the city, its attractions, and customs"""),
            goal=dedent(
                f"""Provide the BEST insights about the selected city"""),
            tools=[SearchTools.search_internet],
            verbose=True,
            llm=llm,
        )