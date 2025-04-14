from typing import Optional, List, Literal
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field, field_validator, ValidationInfo
from uipath import UiPath
from langchain_core.output_parsers import PydanticOutputParser
import logging
import time
from uipath.models import InvokeProcess, IngestionInProgressException
from langchain_core.messages import HumanMessage
from uipath_langchain.retrievers import ContextGroundingRetriever
from langchain_anthropic import ChatAnthropic


logger = logging.getLogger(__name__)

llm = ChatAnthropic(model="claude-3-5-sonnet-latest")

class QuizItem(BaseModel):
    question: str = Field(
        description="One quiz question"
    )
    difficulty: float = Field(
        description="How difficult is the question", ge=0.0, le=1.0
    )
    answer: str = Field(
        description="The expected answer to the question",
    )
class Quiz(BaseModel):
   quiz_items: List[QuizItem] = Field(
        description="A list of quiz items"
    )
class QuizOrInsufficientInfo(BaseModel):
    quiz: Optional[Quiz] = Field(
        description="A quiz based on user input and available documents."
    )
    additional_info: Optional[str] = Field(
        description="String that controls whether additional information is required",
    )

    @field_validator("quiz")
    def check_quiz(cls, v, info: ValidationInfo):
        additional_info = info.data.get("additional_info")
        if additional_info == "false" and v is None:
            raise ValueError("Quiz should be None when additional_info is not 'false'")
        return v

output_parser = PydanticOutputParser(pydantic_object=QuizOrInsufficientInfo)

system_message ="""You are a quiz generator. Try to generate a quiz about {quiz_topic} with multiple questions ONLY based on the following documents. Do not use any extra information from your knowledgebase.
If the documents do not provide enough info, respond with as little words as possible in the format 'additional_info=Need data about ...'. The additional_info should be around 10-15 words.
If they provide enough info, create the quiz and set additional_info='false'

This is the context data: {context}

{format_instructions}

Respond with the classification in the requested JSON format."""

uipath = UiPath()


class GraphOutput(BaseModel):
    quiz: Quiz

class GraphInput(BaseModel):
    quiz_topic: str
    bucket_name: str
    index_name: str
    bucket_folder: Optional[str] = None

class GraphState(MessagesState):
    quiz_topic: str
    bucket_name: str
    bucket_folder: Optional[str]
    index_name: str
    additional_info: Optional[bool]
    quiz: Optional[Quiz]

def prepare_input(state: GraphInput) -> GraphState:
    return GraphState(
        quiz_topic=state.quiz_topic,
        bucket_name=state.bucket_name,
        index_name=state.index_name,
        additional_info="false",
        messages=("user", f"create a quiz about {state.quiz_topic}"),
        bucket_folder=state.bucket_folder,
    )

async def invoke_researcher(state: GraphState) -> Command:
    state["messages"].append(HumanMessage(f"{state['additional_info']}")),

    input_args_json = {
            "messages": state["messages"],
            "bucket_name": state["bucket_name"],
            "bucket_folder": state.get("bucket_folder", None),
        }
    agent_response = interrupt(InvokeProcess(
        name = "researcher-and-uploader-agent",
        input_arguments = input_args_json,
    ))

    return Command(
        update={
            "messages": [agent_response["messages"][-1]],
        })

async def create_quiz(state: GraphState) -> Command:
    no_of_retries = 5
    context_data = None
    data_queried = False
    index = uipath.context_grounding.get_or_create_index(state["index_name"], storage_bucket_name=state["bucket_name"], storage_bucket_folder_path=state["bucket_folder"])
    uipath.context_grounding.ingest_data(index)
    while no_of_retries != 0:
        try:
            context_data = await ContextGroundingRetriever(
                index_name=state["index_name"],
                uipath_sdk=uipath,
                number_of_results=10
            ).ainvoke(state["quiz_topic"])
            data_queried = True
            break
        except IngestionInProgressException as ex:
            logger.info(ex.message)
            no_of_retries -= 1
            logger.info(f"{no_of_retries} retries left")
            time.sleep(5)
    if not data_queried:
        raise Exception("Ingestion is taking too long.")
    message = system_message.format(format_instructions=output_parser.get_format_instructions(),
        context = context_data if context_data else "No context available yet",
        quiz_topic=state["quiz_topic"])
    result = llm.invoke(message)
    try:
        llm_response = output_parser.parse(result.content)
        return Command(
            update={
                "quiz": llm_response.quiz if llm_response.additional_info == "false" else None,
                "additional_info": llm_response.additional_info,
            }
        )
    except Exception as e:
        print(f"Failed to parse {e}")
        return Command(goto=END)

def check_quiz_creation(state: GraphState) -> Literal["invoke_researcher", "return_quiz"]:
    if state["additional_info"] != "false":
        return "invoke_researcher"
    return "return_quiz"

def return_quiz(state: GraphState) -> GraphOutput:
    return GraphOutput(quiz=state["quiz"])

# Build the state graph
builder = StateGraph(input=GraphInput, output=GraphOutput)
builder.add_node("invoke_researcher", invoke_researcher)
builder.add_node("create_quiz", create_quiz)
builder.add_node("return_quiz", return_quiz)
builder.add_node("prepare_input", prepare_input)

builder.add_edge(START, "prepare_input")
builder.add_edge("prepare_input", "create_quiz")
builder.add_conditional_edges("create_quiz", check_quiz_creation)
builder.add_edge("invoke_researcher", "create_quiz")
builder.add_edge("return_quiz", END)

# Compile the graph
graph = builder.compile()
