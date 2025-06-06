# This script uses LangChain to prompt an LLM (OpenAI) to generate a comma-separated list of startup name ideas 
# for a given subject ("AI" in this case). It uses a structured output parser to enforce and extract the list format 
# from the model's raw text output.


from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

output_parser = CommaSeparatedListOutputParser()
format_instructions = output_parser.get_format_instructions()

prompt = PromptTemplate(
    template="Suggest some names for my {subject} startup.\n{format_instructions}",
    input_variables=["subject"],
    partial_variables={"format_instructions": format_instructions}
)

model = OpenAI(temperature=0, openai_api_key=api_key)
_input = prompt.format(subject="AI")
output = model.invoke(_input)
parsed_output = output_parser.parse(output)
print(parsed_output)
