from langchain_core.runnables import RunnableConfig
from langchain_core.pydantic_v1 import BaseModel, Field, create_model
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
)
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import SystemMessage
from langchain_core.language_models import BaseLanguageModel
from langchain_core.documents import Document
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union, cast
import json
import asyncio
from get_prompt import create_unstructured_prompt
prompt1,prompt2 = create_unstructured_prompt()
# print(prompt2)
from pprint import pprint

from graphmemory import GraphMemory, Node, Edge

import json
from openai import OpenAI
import os

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Sample unstructured text
gw_text = "George Washington was the first President of the United States and served from 1789 to 1797."
tj_text = "Thomas Jefferson was the first Secretary of State of the United States and served from 1790 to 1793."
ah_text = "Alexander Hamilton was the first Secretary of the Treasury of the United States and served from 1789 to 1795."


examples = [
    {
        "text": (
            "Adam is a software engineer in Microsoft since 2009, "
            "and last year he got an award as the Best Talent"
        ),
        "head": "Adam",
        "head_type": "Person",
        "relation": "WORKS_FOR",
        "tail": "Microsoft",
        "tail_type": "Company",
    },
    {
        "text": (
            "Adam is a software engineer in Microsoft since 2009, "
            "and last year he got an award as the Best Talent"
        ),
        "head": "Adam",
        "head_type": "Person",
        "relation": "HAS_AWARD",
        "tail": "Best Talent",
        "tail_type": "Award",
    },
    {
        "text": (
            "Microsoft is a tech company that provide "
            "several products such as Microsoft Word"
        ),
        "head": "Microsoft Word",
        "head_type": "Product",
        "relation": "PRODUCED_BY",
        "tail": "Microsoft",
        "tail_type": "Company",
    },
    {
        "text": "Microsoft Word is a lightweight app that accessible offline",
        "head": "Microsoft Word",
        "head_type": "Product",
        "relation": "HAS_CHARACTERISTIC",
        "tail": "lightweight app",
        "tail_type": "Characteristic",
    },
    {
        "text": "Microsoft Word is a lightweight app that accessible offline",
        "head": "Microsoft Word",
        "head_type": "Product",
        "relation": "HAS_CHARACTERISTIC",
        "tail": "accessible offline",
        "tail_type": "Characteristic",
    },
]


def format_example(example):
    return (
        f"Text: {example['text']}\n"
        f"Extracted relation:\n"
        f"{{\"head\": \"{example['head']}\", "
        f"\"head_type\": \"{example['head_type']}\", "
        f"\"relation\": \"{example['relation']}\", "
        f"\"tail\": \"{example['tail']}\", "
        f"\"tail_type\": \"{example['tail_type']}\"}}"
    )

formatted_examples = "\n\n".join(format_example(ex) for ex in examples)

system_prompt = (
    "# Knowledge Graph Instructions for GPT-4\n"
    "## 1. Overview\n"
    "You are a top-tier algorithm designed for extracting information in structured "
    "formats to build a knowledge graph.\n"
    "Try to capture as much information from the text as possible without "
    "sacrificing accuracy. Do not add any information that is not explicitly "
    "mentioned in the text.\n"
    "- **Nodes** represent entities and concepts.\n"
    "- The aim is to achieve simplicity and clarity in the knowledge graph, making it\n"
    "accessible for a vast audience.\n"
    "## 2. Labeling Nodes\n"
    "- **Consistency**: Ensure you use available types for node labels.\n"
    "Ensure you use basic or elementary types for node labels.\n"
    "- For example, when you identify an entity representing a person, "
    "always label it as **'person'**. Avoid using more specific terms "
    "like 'mathematician' or 'scientist'."
    "- **Node IDs**: Never utilize integers as node IDs. Node IDs should be "
    "names or human-readable identifiers found in the text.\n"
    "- **Relationships** represent connections between entities or concepts.\n"
    "Ensure consistency and generality in relationship types when constructing "
    "knowledge graphs. Instead of using specific and momentary types "
    "such as 'BECAME_PROFESSOR', use more general and timeless relationship types "
    "like 'PROFESSOR'. Make sure to use general and timeless relationship types!\n"
    "## 3. Coreference Resolution\n"
    "- **Maintain Entity Consistency**: When extracting entities, it's vital to "
    "ensure consistency.\n"
    'If an entity, such as "John Doe", is mentioned multiple times in the text '
    'but is referred to by different names or pronouns (e.g., "Joe", "he"),'
    "always use the most complete identifier for that entity throughout the "
    'knowledge graph. In this example, use "John Doe" as the entity ID.\n'
    "Remember, the knowledge graph should be coherent and easily understandable, "
    "so maintaining consistency in entity references is crucial.\n"
    "## 4. Strict Compliance\n"
    "Adhere to the rules strictly. Non-compliance will result in termination."
    "## 5. Examples\n"
    "Here are some examples of how to extract information from text:\n\n"
    f"{formatted_examples}\n\n"
    "Follow these examples when extracting information from the given text."
)

# print(system_prompt)
node_labels = None
rel_types = None
# Extract structured data from unstructured text
human_string_parts = [
    "For the following text, extract entities and relations as "
    "in the provided example.\nText: {input}",
]


class UnstructuredRelation(BaseModel):
    head: str = Field(
        description=(
            "extracted head entity like Microsoft, Apple, John. "
            "Must use human-readable unique identifier."
        )
    )
    head_type: str = Field(
        description="type of the extracted head entity like Person, Company, etc"
    )
    relation: str = Field(
        description="relation between the head and the tail entities")
    tail: str = Field(
        description=(
            "extracted tail entity like Microsoft, Apple, John. "
            "Must use human-readable unique identifier."
        )
    )
    tail_type: str = Field(
        description="type of the extracted tail entity like Person, Company, etc"
    )


parser = JsonOutputParser(pydantic_object=UnstructuredRelation)
human_prompt_string = "\n".join(filter(None, human_string_parts))
human_prompt = PromptTemplate(
    template=human_prompt_string,
    input_variables=["input"],
    # partial_variables={
    #     "format_instructions": parser.get_format_instructions(),
    #     "node_labels": node_labels,
    #     "rel_types": rel_types,
    #     "examples": examples,
    # },
)

human_message_prompt = HumanMessagePromptTemplate(prompt=human_prompt)

def create_openai_messages2(input_text: str) -> List[Dict[str, str]]:
    # print("HUMAN PROMPT",human_prompt)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": human_prompt.format(input=input_text)}
    ]

my_prompt = create_openai_messages2('HELLO')
# pprint(my_prompt)
# print("")
# print("")
# print("")

def extract_attributes(text):
    prompt = create_openai_messages2(text)
    # print(prompt)
    response =  client.chat.completions.create(
        model="gpt-4o-mini",
        # messages=[
        #     {"role": "system", "content": "Extract structured data from this text using the following attributes: \
        #      name, title, country, term_start, term_end"},
        #     {"role": "user", "content": text}
        # ],
        messages = prompt,
        seed=1
    )
    print(response)
    print(response.choices[0].message.content)

# Calculate embedding for a given input


def calculate_embedding(input_json):
    return client.embeddings.create(
        input=input_json,
        model="text-embedding-3-small"
    ).data[0].embedding


gw_embedding = calculate_embedding(gw_text)
tj_embedding = calculate_embedding(tj_text)
ah_embedding = calculate_embedding(ah_text)

# Initialize the database from disk (make sure to set vector_length correctly)
graph_db = GraphMemory(database='graph.db', vector_length=len(gw_embedding))

# Extract structured data from unstructured text
gw_attributes = extract_attributes(gw_text)
tj_attributes = extract_attributes(tj_text)
ah_attributes = extract_attributes(ah_text)

print(gw_attributes)
print(tj_attributes)
print(ah_attributes)

# Output Example:
# {
#   'person': 'George Washington',
#   'title': 'President',
#   'country': 'United States',
#   'term_start': '1789',
#   'term_end': '1797'
# }
# {
#   'person': 'Thomas Jefferson',
#   'title': 'Secretary of State',
#   'country': 'United States',
#   'term_start': 1790,
#   'term_end': 1793
# }
# {
#   'person': 'Alexander Hamilton',
#   'title': 'Secretary of the Treasury',
#   'country': 'United States',
#   'term_start': 1789,
#   'term_end': 1795
# }


# Create nodes with UUIDs
gw_node = Node(properties=gw_attributes, vector=gw_embedding)
tj_node = Node(properties=tj_attributes, vector=tj_embedding)
ah_node = Node(properties=ah_attributes, vector=ah_embedding)

gw_node_id = graph_db.insert_node(gw_node)
if gw_node_id is None:
    raise ValueError("Failed to insert George Washington node")

tj_node_id = graph_db.insert_node(tj_node)
if tj_node_id is None:
    raise ValueError("Failed to insert Thomas Jefferson node")

ah_node_id = graph_db.insert_node(ah_node)
if ah_node_id is None:
    raise ValueError("Failed to insert Alexander Hamilton node")

# Insert edges
edge1 = Edge(source_id=gw_node_id, target_id=tj_node_id,
             relation="served_under", weight=0.5)
edge2 = Edge(source_id=gw_node_id, target_id=ah_node_id,
             relation="served_under", weight=0.5)
graph_db.insert_edge(edge1)
graph_db.insert_edge(edge2)

# Print edges
print(graph_db.edges_to_json())

# Find connected nodes
connected_nodes = graph_db.connected_nodes(gw_node_id)
for node in connected_nodes:
    print("Connected Node Data:", node.properties)

# Find nearest nodes by vector embedding
nearest_nodes = graph_db.nearest_nodes(
    calculate_embedding("George Washington"), limit=1)
# print(nearest_nodes)
print("Nearest Node Data:", nearest_nodes[0].node.properties)
print("Nearest Node Distance:", nearest_nodes[0].distance)

# Get node/s by attribute (Who was the Secretary of State?)
nodes = graph_db.nodes_by_attribute("title", "Secretary of State")
if nodes:
    print("Node by attribute:", nodes[0].properties)
else:
    print("No nodes found with the attribute 'title' = 'Secretary of State'")

# What is the title of the people who served under George Washington?
for node in connected_nodes:
    print(f"{node.properties.get('name')} - {node.properties.get('title')}")

# Fetch a node by UUID
fetched_node = graph_db.get_node(gw_node_id)

# Delete an edge by source / target node id
graph_db.delete_edge(edge1.source_id, edge1.target_id)
