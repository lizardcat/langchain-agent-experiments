"""
Very simple agent that uses OpenAI API and OpenWeather API to deliver weather information about a particular location.
"""


import os
from pyowm.owm import OWM
from pyowm.utils.config import get_default_config
from typing import Type
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
weather_api_key = os.getenv("WEATHER_API_KEY")

def get_weather(city):
    config_dict = get_default_config()
    config_dict['language'] = 'en' # your language here
    owm = OWM(weather_api_key, config_dict)
    mgr = owm.weather_manager()
    observation = mgr.weather_at_place("Nairobi, Kenya")

    temperature=str(observation.weather.temperature("celsius")["temp"]) + '°C'
    humidity = str(observation.weather) + '%'
    wind = str(observation.weather.wind()) + ' m/s'
    status = observation.weather.detailed_status

    return{'temp':temperature, 'humid':humidity, 'wind':wind, 'status':status}

class GetWeatherInput(BaseModel):
    """Inputs for get_weather"""
    city:str=Field(description="City name with country separated by comma")

class GetWeatherTool(BaseTool):
    name: str = "get_weather_details_of_a_city"
    description: str = """
        Useful to get weather details of a city. 
        Mandatory input format is 'city, country'.
        """
    args_schema: Type[BaseModel] = GetWeatherInput

    def _run(self, city: str):
        weather = get_weather(city)
        return weather

    def _arun(self, city: str):
        raise NotImplementedError("this tool doesn't support async")
    
llm = ChatOpenAI(openai_api_key=api_key)
tools = [GetWeatherTool()]

model = ChatOpenAI(openai_api_key=api_key)
planner = load_chat_planner(model)
executor = load_agent_executor(model, tools, verbose=True)
agent=PlanAndExecute(planner=planner, executor=executor, verbose=True)

agent.run('Should I visit Nairobi, Kenya, this December based on the weather conditions?')
