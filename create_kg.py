# first = ChatPromptTemplate(input_variables=['input'], messages=[SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template='# Knowledge Graph Instructions for GPT-4\n## 1. Overview\nYou are a top-tier algorithm designed for extracting information in structured formats to build a knowledge graph.\nTry to capture as much information from the text as possible without sacrificing accuracy. Do not add any information that is not explicitly mentioned in the text.\n- **Nodes** represent entities and concepts.\n- The aim is to achieve simplicity and clarity in the knowledge graph, making it\naccessible for a vast audience.\n## 2. Labeling Nodes\n- **Consistency**: Ensure you use available types for node labels.\nEnsure you use basic or elementary types for node labels.\n- For example, when you identify an entity representing a person, always label it as **\'person\'**. Avoid using more specific terms like \'mathematician\' or \'scientist\'.- **Node IDs**: Never utilize integers as node IDs. Node IDs should be names or human-readable identifiers found in the text.\n- **Relationships** represent connections between entities or concepts.\nEnsure consistency and generality in relationship types when constructing knowledge graphs. Instead of using specific and momentary types such as \'BECAME_PROFESSOR\', use more general and timeless relationship types like \'PROFESSOR\'. Make sure to use general and timeless relationship types!\n## 3. Coreference Resolution\n- **Maintain Entity Consistency**: When extracting entities, it\'s vital to ensure consistency.\nIf an entity, such as "John Doe", is mentioned multiple times in the text but is referred to by different names or pronouns (e.g., "Joe", "he"),always use the most complete identifier for that entity throughout the knowledge graph. In this example, use "John Doe" as the entity ID.\nRemember, the knowledge graph should be coherent and easily understandable, so maintaining consistency in entity references is crucial.\n## 4. Strict Compliance\nAdhere to the rules strictly. Non-compliance will result in termination.')), HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template='Tip: Make sure to answer in the correct format and do not include any explanations. Use the given format to extract information from the following input: {input}'))]) middle = [{
#   raw: RunnableBinding(bound=ChatOpenAI(client= <openai.resources.chat.completions.Completions object at 0x17cb548e0>, async_client=<openai.resources.chat.completions.AsyncCompletions object at 0x17cb55ff0>, model_name='gpt-4-turbo', temperature=0.0, openai_api_key=SecretStr('**********'), openai_proxy=''), kwargs={'tools': [{'type': 'function', 'function': {'name': 'DynamicGraph', 'description': 'Represents a graph document consisting of nodes and relationships.', 'parameters': {'type': 'object', 'properties': {'nodes': {'description': 'List of nodes', 'type': 'array', 'items': {'type': 'object', 'properties': {'id': {'description': 'Name or human-readable unique identifier.', 'type': 'string'}, 'type': {'description': "The type or label of the node.Ensure you use basic or elementary types for node labels.\nFor example, when you identify an entity representing a person, always label it as **'Person'**. Avoid using more specific terms like 'Mathematician' or 'Scientist'", 'type': 'string'}}, 'required': ['id', 'type']}}, 'relationships': {'description': 'List of relationships', 'type': 'array', 'items': {'type': 'object', 'properties': {'source_node_id': {'description': 'Name or human-readable unique identifier of source node', 'type': 'string'}, 'source_node_type': {'description': "The type or label of the source node.Ensure you use basic or elementary types for node labels.\nFor example, when you identify an entity representing a person, always label it as **'Person'**. Avoid using more specific terms like 'Mathematician' or 'Scientist'", 'type': 'string'}, 'target_node_id': {'description': 'Name or human-readable unique identifier of target node', 'type': 'string'}, 'target_node_type': {'description': "The type or label of the target node.Ensure you use basic or elementary types for node labels.\nFor example, when you identify an entity representing a person, always label it as **'Person'**. Avoid using more specific terms like 'Mathematician' or 'Scientist'", 'type': 'string'}, 'type': {'description': "The type of the relationship.Instead of using specific and momentary types such as 'BECAME_PROFESSOR', use more general and timeless relationship types like 'PROFESSOR'. However, do not sacrifice any accuracy for generality", 'type': 'string'}}, 'required': ['source_node_id', 'source_node_type', 'target_node_id', 'target_node_type', 'type']}}}}}}], 'parallel_tool_calls': False, 'tool_choice': {'type': 'function', 'function': {'name': 'DynamicGraph'}}})
from typing import Any, Dict
from pprint import pprint
import json
from typing import cast
from openai import OpenAI
client = OpenAI()

input_text = '''
Context: [PAR] [TLE] Robert Sheehan [SEP] Robert Michael Sheehan (Irish: "Roibeárd Mícheál Ó Siodhacháin" ; born 7 January 1988) is an Irish actor.  He is best known for television roles such as Nathan Young in "Misfits" and Darren in "Love/Hate", as well as the 2009 film "Cherrybomb" alongside Rupert Grint.  He also co-starred in the film "Killing Bono" as Ivan McCormick.  In late 2011 he starred in John Crowley's production of J. M. Synge's comic play "The Playboy of the Western World" at the Old Vic Theatre in London. [PAR] [TLE] The Messenger (2015 horror film) [SEP] The Messenger is a 2015 British supernatural mystery horror film directed by David Blair, written by Andrew Kirk and starring Robert Sheehan and Lily Cole.'''

node_json_object = json.dumps({
    "nodes": {
        "description": "List of nodes",
        "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "description": "Name or human-readable unique identifier.",
                            "type": "string"
                        },
                        "type": {
                            "description": "The type or label of the node. Ensure you use basic or elementary types for node labels.\nFor example, when you identify an entity representing a person, always label it as **'Person'**. Avoid using more specific terms like 'Mathematician' or 'Scientist'",
                            "type": "string"
                        }
                    },
                    "required": ["id", "type"]
                }
    },
})

relationship_json_object = json.dumps({
    "relationships": {
        "description": "List of relationships",
        "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source_node_id": {
                            "description": "Name or human-readable unique identifier of source node",
                            "type": "string"
                        },
                        "source_node_type": {
                            "description": "The type or label of the source node. Ensure you use basic or elementary types for node labels.\nFor example, when you identify an entity representing a person, always label it as **'Person'**. Avoid using more specific terms like 'Mathematician' or 'Scientist'",
                            "type": "string"
                        },
                        "target_node_id": {
                            "description": "Name or human-readable unique identifier of target node",
                            "type": "string"
                        },
                        "target_node_type": {
                            "description": "The type or label of the target node. Ensure you use basic or elementary types for node labels.\nFor example, when you identify an entity representing a person, always label it as **'Person'**. Avoid using more specific terms like 'Mathematician' or 'Scientist'",
                            "type": "string"
                        },
                        "type": {
                            "description": "The type of the relationship. Instead of using specific and momentary types such as 'BECAME_PROFESSOR', use more general and timeless relationship types like 'PROFESSOR'. However, do not sacrifice any accuracy for generality",
                            "type": "string"
                        }
                    },
                    "required": ["source_node_id", "source_node_type", "target_node_id", "target_node_type", "type"]
                }
    }

})

sample_nodes = json.dumps({
    "nodes": [
        {"id": "node1", "type": "Person"},
        {"id": "node2", "type": "Organization"},
        {"id": "node3", "type": "Person"},
        {"id": "node4", "type": "Country"},
        {"id": "node5", "type": "Age"},
    ],
})

sample_relationships = json.dumps({
    "relationships": [
        {"source_node_id": "node1", "source_node_type": "Person", "target_node_id": "node2",
         "target_node_type": "Organization", "type": "WORKS_FOR"},
        {"source_node_id": "node3", "source_node_type": "Person", "target_node_id": "node2",
         "target_node_type": "Organization", "type": "MEMBER_OF"}
    ]
})

#CONTAINS ENTITY EXTRACTION PROMPT
#region
extracting_entities = f"""
# Knowledge Graph Instructions for GPT-4: Entity Extraction

## 1. Overview
You are a top-tier algorithm designed for extracting information in structured formats to build a knowledge graph. Your task is to identify and label all entities from the given text.

- **Nodes** represent entities and concepts. Extract as many entities as possible without sacrificing accuracy.
- **Basic Information Extraction**: Include fundamental details such as names, dates of birth, age, nationality, etc. For instance, if the text mentions a person's birthdate or a dog's breed, include these details as part of the entities.
- **Criteria**: Aim to extract as many entities as possible that may be relevant. This should include any Nouns, Adjectives, Pronouns, Proper Nouns, etc. 

## 2. Labeling Nodes
- **Consistency**: Use basic or elementary types for node labels.
- Label entities according to their most general type (e.g., 'person', 'place', 'organization'). Avoid specific labels like 'mathematician' or 'scientist'.
- **Node IDs**: Use human-readable identifiers found in the text as Node IDs. Do not use integers.

## 3. Coreference Resolution
- **Maintain Entity Consistency**: Use the most complete identifier for an entity throughout the text. For example, if "John Doe" is mentioned multiple times, use "John Doe" as the entity ID in your output.

## 4. Strict Compliance
Follow these instructions strictly. Non-compliance will result in termination.

## 5. JSON Output Format
- **Output Format**: Provide your response in the following JSON format for entities:
- **Maintain JSON Output Format** You must reply in the following JSON format:
{node_json_object}
Here is an example response:
{sample_nodes}
```
"""
#endregion

#CONTAINS RELATIONSHIP EXTRACTION PROMPT
#region
relationship_prompt = f"""
# Knowledge Graph Instructions for GPT-4: Relationship Extraction

# 1. Overview
You are a top-tier algorithm designed for extracting relationships between entities to build a knowledge graph. Your task is to identify and label all relationships between the entities extracted from the given text.

- **Relationships ** represent connections between entities or concepts. Ensure consistency and generality in relationship types.
- Use general and timeless relationship types(e.g., 'PROFESSOR', 'FAMILY_MEMBER'). Avoid specific and momentary types like 'BECAME_PROFESSOR'.
- **Basic Information Extraction**: In addition to complex relationships, extract fundamental details such as the Age of a Person, the Nationality of a Person, the Alias of an Organization, the founding year of a Company, the Species of a Dog, etc. 

# 2. Relationship Construction
- **Consistency**: Ensure you use consistent types for relationships throughout the knowledge graph.
- **Node IDs**: Use the entity IDs identified in the previous step to construct relationships.

# 3. Strict Compliance
Follow these instructions strictly. Non-compliance will result in termination.

# 4. JSON Output Format
- **Output Format**: Provide your response in the following JSON format for relationships:
{relationship_json_object}
Here is an example response:
{sample_relationships}
```
"""
#endregion


def validate_entities(entities: Dict[str, Any]) -> bool:
    try:
        # Check the top-level structure
        if "nodes" not in entities or not isinstance(entities["nodes"], list):
            return False

        # Validate each node
        for node in entities["nodes"]:
            if not isinstance(node, dict):
                return False
            if "id" not in node or "type" not in node:
                return False
            if not isinstance(node["id"], str) or not isinstance(node["type"], str):
                return False

        return True
    except Exception as e:
        print(f"Validation error: {e}")
        return False


def validate_relationships(relationships: Dict[str, Any]) -> bool:
    try:
        # Check the top-level structure
        if "relationships" not in relationships or not isinstance(relationships["relationships"], list):
            print("its this")
            return False

        # Validate each relationship
        for relationship in relationships["relationships"]:
            if not isinstance(relationship, dict):
                print("no its this")
                return False
            if "source_node_id" not in relationship or \
               "target_node_id" not in relationship or \
               "type" not in relationship:
                print(relationship)
                print("oh no its this actually")
                return False
            if not isinstance(relationship["source_node_id"], str) or \
               not isinstance(relationship["target_node_id"], str) or \
               not isinstance(relationship["type"], str):
                print('here')
                return False

        return True
    except Exception as e:
        print(f"Validation error: {e}")
        return False

# Define the Node class


class Node:
    def __init__(self, id: str, type: str):
        self.id = id
        self.type = type

    def __repr__(self):
        return f"Node(id={self.id}, type={self.type})"

# Define the Relationship class

class Relationship:
    def __init__(self, source_node_id: str, source_node_type: str, target_node_id: str, target_node_type: str, type: str):
        self.source = Node(id=source_node_id, type=source_node_type)
        self.target = Node(id=target_node_id, type=target_node_type)
        self.type = type

    def __repr__(self):
        return f"Relationship(source_node_id={self.source.id}, source_node_type={self.source.type}, target_node_id={self.target.id}, target_node_type={self.target.type}, type={self.type})"

# Function to convert JSON to Node objects

def json_to_nodes(json_data):
    nodes = []
    for node in json_data["nodes"]:
        nodes.append(Node(id=node["id"], type=node["type"]))
    return nodes

# Function to convert JSON to Relationship objects


def json_to_relationships(json_data):
    relationships = []
    for relationship in json_data["relationships"]:
        relationships.append(Relationship(
            source_node_id=relationship["source_node_id"],
            source_node_type=relationship.get("source_node_type","unknown"),
            target_node_id=relationship["target_node_id"],
            target_node_type=relationship.get("target_node_type","unknown"),
            type=relationship["type"]
        ))
    return relationships


# Example usage:
nodes_json = {
    "nodes": [
        {"id": "node1", "type": "Person"},
        {"id": "node2", "type": "Organization"},
    ]
}

relationships_json = {
    "relationships": [
        {"source_node_id": "node1", "source_node_type": "Person", "target_node_id": "node2",
            "target_node_type": "Organization", "type": "WORKS_FOR"}
    ]
}

def create_graph(input_text: str): 
    entities = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": extracting_entities
            },
            {
                "role": "user",
                "content": f"Tip: Make sure to answer in the correct format and do not include any explanations. Use the given format to extract information from the following input: {input_text}"
            }
        ],
        model="gpt-4o-mini",
        response_format={"type": "json_object"},

    )
    entities = json.loads(entities.choices[0].message.content)
    print("ENTITIES: ", entities)
    relationships = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": relationship_prompt
            },
            {
                "role": "user",
                "content": f"Tip: Make sure to answer in the correct format and do not include any explanations. Use the given format to extract information from the following input: {input_text}. Extracted Entities: {entities}"
            }
        ],
        model="gpt-4o-mini",
        response_format={"type": "json_object"},

    )
    # print(response)
    relationships = json.loads(relationships.choices[0].message.content)

    if not validate_entities(entities):
        raise ValueError("Invalid entities format")
    if not validate_relationships(relationships):
        raise ValueError("Invalid relationships format")
    
    entities = json_to_nodes(entities)
    relationships = json_to_relationships(relationships)
    return entities, relationships


# raw_schema = cast(Dict[Any, Any], response)
# nodes, relationships = _convert_to_graph_document(raw_schema)
