from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from graphmemory import GraphMemory, Node, Edge

import json
from openai import OpenAI
import os

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Sample unstructured text
gw_text = "George Washington was the first President of the United States and served from 1789 to 1797."
tj_text = "Thomas Jefferson was the first Secretary of State of the United States and served from 1790 to 1793."
ah_text = "Alexander Hamilton was the first Secretary of the Treasury of the United States and served from 1789 to 1795."

# Extract structured data from unstructured text

# def extract_attributes(text):
#     return json.loads(client.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             {"role": "system", "content": "Extract structured data from this text using the following attributes: \
#              name, title, country, term_start, term_end"},
#             {"role": "user", "content": text}
#         ],
#         seed=1
#     ).choices[0].message.content)

# Calculate embedding for a given input

#Question: Calculate the embedding for the input or for the triplet?
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
# gw_attributes = extract_attributes(gw_text)
# tj_attributes = extract_attributes(tj_text)
# ah_attributes = extract_attributes(ah_text)

# print(gw_attributes)
# print(tj_attributes)
# print(ah_attributes)

# # Create nodes with UUIDs
# gw_node = Node(properties=gw_attributes, vector=gw_embedding)
# tj_node = Node(properties=tj_attributes, vector=tj_embedding)
# ah_node = Node(properties=ah_attributes, vector=ah_embedding)


# gw_node_id = graph_db.insert_node(gw_node)
# if gw_node_id is None:
#     raise ValueError("Failed to insert George Washington node")

# tj_node_id = graph_db.insert_node(tj_node)
# if tj_node_id is None:
#     raise ValueError("Failed to insert Thomas Jefferson node")

# ah_node_id = graph_db.insert_node(ah_node)
# if ah_node_id is None:
#     raise ValueError("Failed to insert Alexander Hamilton node")


# Insert edges
# edge1 = Edge(source_id=gw_node_id, target_id=tj_node_id,
#              relation="served_under", weight=0.5)
# edge2 = Edge(source_id=gw_node_id, target_id=ah_node_id,
#              relation="served_under", weight=0.5)
# graph_db.insert_edge(edge1)
# graph_db.insert_edge(edge2)

# # Print edges

texts = [
    "George Washington was the first President of the United States and served from 1789 to 1797.",
    "Thomas Jefferson was the first Secretary of State of the United States and served from 1790 to 1793.",
    "Alexander Hamilton was the first Secretary of the Treasury of the United States and served from 1789 to 1795."
]

llm = ChatOpenAI(temperature=0, model_name="gpt-4-turbo")

llm_transformer = LLMGraphTransformer(llm=llm)
def process_text(text):
    documents = [Document(page_content=text)]
    graph_documents = llm_transformer.convert_to_graph_documents(documents)
    nodes = graph_documents[0].nodes
    edges = graph_documents[0].relationships
    for node in nodes:
        embedding = client.embeddings.create(
            input=node.id, model="text-embedding-3-small").data[0].embedding
        graph_node = Node(
            properties={"id": node.id, "type": node.type}, vector=embedding)
        node_id = graph_db.insert_node(graph_node)
        if node_id is None:
            raise ValueError(f"Failed to insert node: {node.id}")

    for rel in edges:
        # source_id = graph_db.get_node_by_property("id", rel.source.id).uuid
        source_id = graph_db.get_node(rel.source).uuid
        # target_id = graph_db.get_node_by_property("id", rel.target.id).uuid
        target_id = graph_db.get_node(rel.target).uuid
        edge = Edge(source_id=source_id, target_id=target_id,
                    relation=rel.type, weight=0.5)
        graph_db.insert_edge(edge)

# Process all texts
for text in texts:
    process_text(text)
    
    
print(graph_db.edges_to_json())

# Find connected nodes
connected_nodes = graph_db.connected_nodes(gw_node_id)
for node in connected_nodes:
    print("Connected Node Data:", node.properties)

# Find nearest nodes by vector embedding
nearest_nodes = graph_db.nearest_nodes(
    calculate_embedding("George Washington"), limit=1)
print(nearest_nodes)
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
