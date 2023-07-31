from typing import Any
from langchain.tools import Tool, BaseTool
from langchain.tools.sql_database.tool import BaseSQLDatabaseTool
from pydantic import BaseModel, Field, validator, root_validator
from langchain import LLMChain, PromptTemplate
from langchain.llms import OpenAI
from utils.prompts import QUERY_CHECKER

class QuerySQLDatabaseTool(BaseSQLDatabaseTool, BaseTool):
    """Tool for querying a SQL database"""

    name = "query_sql_database"
    description = """
    Input to this tools is a detailed and correct SQL query, output is a result from the database.
    If the query is not correct, the tool will return an error message.
    If an error is returned, rewrite the query and try again.
    """

    def _run(self, query: str) -> str:
        """Run the tool"""
        return self.db.run_no_throw(query)
    async def _arun(self, query: str) -> str:
        """Run the tool"""
        return self.db.run_no_throw(query)
    
class InfoSQLDatabaseTool(BaseSQLDatabaseTool, BaseTool):
    """Tool for getting metadata about a SQL database"""

    name = "schema_sql_database"
    description = """
    Input to this tools is a comma-separated list of table names, output is schema and sample data from those tables.
    Example Input: "table1, table2, table3"
    """

    def _run(self, tables: str) -> str:
        """Run the tool"""
        return self.db.get_table_info_no_throw(tables.split(","))
    async def _arun(self, tables: str) -> str:
        """Run the tool"""
        return self.db.get_table_info_no_throw(tables.split(","))
    
class ListSQLDatabaseTool(BaseSQLDatabaseTool, BaseTool):
    """Tool for listing tables in a SQL database"""

    name = "list_sql_database"
    description = """Input is an empty string, output is a list of tables in the database."""

    def _run(self, query: str) -> str:
        return ", ".join(self.db.get_usable_table_names())
    async def _arun(self, query: str) -> str:
        return ", ".join(self.db.get_usable_table_names())

def FormatBigNumbers(BaseTool):
    """Tool for formatting big numbers in thousand, million, billion, and trillion"""

    name = "format_big_numbers"
    description = """Input is a number, output is the number formatted in thousand, million, billion, and trillion."""
    format_lookup = {
        1_000: "thousand",
        1_000_000: "million",
        1_000_000_000: "billion",
        1_000_000_000_000: "trillion",
    }

    def _run(self, tool_input: str = "") -> str:
        """Format the number"""
        try:
            numbers = tool_input.split(", ")
            for i in range(len(numbers)):
                x = int(numbers[i].replace(",", ""))
                for k in format_lookup.keys():
                    if abs(x // k) > 0:
                        numbers[i] = f"{x/k} {format_lookup[k]}"
                    else:
                        break
            return numbers
        except Exception as e:
            return tool_input
    
    async def _arun(self, tool_input):
        return _run(tool_input)
    
class QueryCheckerTool(BaseSQLDatabaseTool, BaseTool):
    """Use an LLM to check if a query is correct.
    Adapted from https://www.patterns.app/blog/2023/01/18/crunchbot-sql-analyst-gpt/
    """

    template: str = QUERY_CHECKER
    llm_chain = Field(
        default_factory=lambda: LLMChain(
            llm = OpenAI(
                model_name = "gpt-3.5-turbo",
                temperature=0,
                model_kwargs={
                    'engine': 'deploy-oai-use-chatGPT-01-text-davinci-003',
            }),
            prompt = PromptTemplate(
                template,
                input_variables=["query", "dialect"]
            ),
        )
    )
    name = "query_checker_sql_database"
    description = """
    Use this tool to double check if your SQL query is correct before running it.
    Always use this tool before running a query with the "query_sql_database" tool.
    """

    @validator("llm_chain")
    def validate_llm_chain_input_variables(cls, llm_chain):
        """Validate that the LLMChain input variables are correct"""
        if llm_chain.prompt.input_variables != ["query", "dialect"]:
            raise ValueError(
                "LLMChain input variables must be ['query', 'dialect']"
            )
        return llm_chain
    
    def _run(self, query: str) -> str:
        return self.llm_chain.predict(query, dialect=self.db.dialect)
    
    async def _arun(self, query: str) -> str:
        return self.llm_chain.apredict(query, dialect=self.db.dialect)
