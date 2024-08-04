import os
import pickle
import networkx as nx
import torch
from scipy.spatial.distance import cosine
from transformers import BertTokenizer, BertModel
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
import openai
from openai import OpenAI
import numpy as np


class Node:
    def __init__(self, id, node_type):
        self.id = id
        self.type = node_type


class Relationship:
    def __init__(self, source, target, relationship_type):
        self.source = source  # Instance of Node
        self.target = target  # Instance of Node
        self.type = relationship_type

# Initialize LLM
llm = ChatOpenAI(temperature=0, model_name="gpt-4-turbo")
llm_transformer = LLMGraphTransformer(llm=llm)

# Initialize BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def clear_graph(graph):
    """
    Clears all nodes and edges from the given graph.
    """
    graph.clear()
    print("Graph has been cleared.")
    
def load_graph(filename='graph.pkl'):
    """
    Loads the graph from a file using pickle.
    """
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    else:
        return nx.DiGraph()


def save_graph(graph, filename='graph.pkl'):
    """
    Saves the graph to a file using pickle.
    """
    with open(filename, 'wb') as f:
        pickle.dump(graph, f, pickle.HIGHEST_PROTOCOL)


def process_text_to_graph(text):
    """
    Converts text to a graph using LLM-based transformation.
    """
    documents = [Document(page_content=text)]
    graph_documents = llm_transformer.convert_to_graph_documents(documents)
    return graph_documents[0].nodes, graph_documents[0].relationships


def create_node_edge_representation(graph, node_id):
    """
    Creates a textual representation of a node and its edges.
    """
    node_data = graph.nodes[node_id]
    descriptions = [f"{node_id}"]
    for target_id in graph.neighbors(node_id):
        edge_type = graph[node_id][target_id].get('type', 'UNKNOWN')
        descriptions.append(f"{edge_type} {target_id}")
    return ' '.join(descriptions)


def encode_text(text):
    """
    Encodes text into BERT embeddings.
    """
    inputs = tokenizer(text, return_tensors='pt',
                       truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def get_text_representation(relationship):
    """
    Creates a detailed textual representation for a relationship.
    
    Args:
        relationship (Relationship): The relationship object containing source, target, and type.
        
    Returns:
        str: Detailed text representation.
    """
    source_node = relationship.source.id
    source_type = relationship.source.type
    target_node = relationship.target.id
    target_type = relationship.target.type
    relationship_type = relationship.type

    # return (f"Source Node: {source_node} (Type: {source_type}), "
    #         f"Relationship Type: {relationship_type}, "
    #         f"Target Node: {target_node} (Type: {target_type}), "
    return(f"Description: This relationship indicates that {source_node}, a {source_type} has a {relationship_type} towards {target_node}, a {target_type}.")


def get_all_relationship_texts(graph):
    """
    Retrieves the text representation for every relationship in the graph.
    
    Args:
        graph (NetworkX graph): The graph containing relationships as edges.
        
    Returns:
        dict: A dictionary where keys are tuples of (source_node, target_node) and values are text representations of the relationships.
    """
    relationship_texts = {}

    # Iterate through all edges in the graph
    for source, target, data in graph.edges(data=True):
        # Assuming 'data' contains the relationship data with 'type' and nodes have 'id' and 'type'
        relationship = {
            'source': graph.nodes[source],
            'target': graph.nodes[target],
            # Default to 'unknown' if type is not provided
            'type': data.get('type', 'unknown')
        }

        # Get the text representation of the relationship
        text_representation = get_text_representation(relationship)

        # Store the text representation in the dictionary
        relationship_texts[(source, target)] = text_representation

    return relationship_texts

def find_similar_relationships(graph, new_relationship, threshold=0.8):
    """
    Finds existing relationships in the graph that are similar to a new relationship.
    Returns a list of tuples containing the source node, target node, existing relationship data, and similarity score.
    """
    similar_relationships = []
    new_text = get_text_representation(new_relationship)
    new_embedding = encode_text(new_text)

    for source, target, data in graph.edges(data=True):
        source_node_data = graph.nodes[source]
        target_node_data = graph.nodes[target]

        # Create Node objects
        source_node = Node(
            id=source,
            node_type=source_node_data.get('type', 'unknown')
        )
        target_node = Node(
            id=target,
            node_type=target_node_data.get('type', 'unknown')
        )

        # Create Relationship object
        relationship = Relationship(
            source=source_node,
            target=target_node,
            relationship_type=data.get('type', 'unknown')
        )
        # Get the text representation of the relationship
        existing_text = get_text_representation(relationship)
        print("EXISTING TEXT ",existing_text)
        print("NEW TEXT ",new_text)
        existing_embedding = encode_text(existing_text)
        similarity = 1 - cosine(new_embedding, existing_embedding)
        print("SIMILARITY ",similarity)
        if similarity > threshold:
            similar_relationships.append((source, target, data, similarity))

    return similar_relationships


def resolve_conflict_with_llm(existing_relationship, new_relationship):
    """
    Resolves conflicts between an existing relationship and a new relationship using an LLM.
    
    Args:
        existing_relationship (Relationship): The existing relationship.
        new_relationship (Relationship): The new relationship.
        
    Returns:
        str: Action to take (e.g., 'update', 'merge', 'discard').
    """
    existing_text = get_text_representation(existing_relationship)
    new_text = get_text_representation(new_relationship)


    prompt = (
        "We have two similar relationships in a knowledge graph. Here are their descriptions:\n\n"
        f"Existing Relationship: {existing_text}\n\n"
        f"New Relationship: {new_text}\n\n"
        "Please determine the appropriate action to take. Choose from one of the following options and provide a brief explanation:\n\n"
        "1. **Update**: If the new relationship provides more recent or accurate information that should replace the existing relationship. Example: If the existing relationship states 'John's favourite ice cream is vanilla' and the new relationship states 'John's favourite ice cream is chocolate,' the action should be 'update' because the new information is more current.\n\n"
        "2. **Merge**: If the new relationship adds valuable information to the existing relationship without contradicting it. Example: If the existing relationship states 'John likes ice cream' and the new relationship adds 'John's favourite ice cream is chocolate,' the action should be 'merge' because the new information complements the existing one.\n\n"
        "3. **Discard**: If the new relationship does not add any new value or is redundant. Example: If the existing relationship and the new relationship both state 'John's favourite ice cream is chocolate,' the action should be 'discard' because the new relationship does not provide additional information.\n\n"
        "4. **Not Connected**: If the existing relationship and the new relationship are unrelated or refer to different entities or contexts. Example: If the existing relationship is about 'John's favourite ice cream is vanilla' and the new relationship is 'John went to the store,' the action should be 'not connected' because they do not pertain to the same context or entities.\n\n"
        "Please respond with one of the four options: 'update,' 'merge,' 'discard,' or 'not-connected,'. Respond with only one word."
    )

    client = OpenAI()
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-4o-mini",
    )
    # response = llm(prompt)
    print("1) ",existing_text,"\n 2) ",new_text,"\n 3)", response.choices[0].message.content)
    # print('RESPONSE ', response['choices'][0]['message'])
    return response.choices[0].message.content


def update_graph_with_new_data(graph, nodes, relationships):
    """
    Updates the graph with new nodes and relationships, removing duplicates.
    """
    processed_relationships = set()

    # Add nodes to the graph with attributes
    for node in nodes:
        if not graph.has_node(node.id):
            graph.add_node(node.id, type=node.type)

    # Check and update relationships
    for relationship in relationships:
        # Skip if relationship has already been processed
        if (relationship.source.id, relationship.target.id, relationship.type) in processed_relationships:
            continue

        similar_relationships = find_similar_relationships(graph, relationship)

        if not similar_relationships:
            graph.add_edge(relationship.source.id,
                           relationship.target.id, type=relationship.type)
        else:
            # Resolve conflicts using LLM
            for similar in similar_relationships:
                existing_relationship = Relationship(
                    source=Node(id=similar[0], node_type=graph.nodes[similar[0]].get(
                        'type', 'unknown')),
                    target=Node(id=similar[1], node_type=graph.nodes[similar[1]].get(
                        'type', 'unknown')),
                    relationship_type=similar[2].get('type', 'unknown')
                )
                action = resolve_conflict_with_llm(
                    existing_relationship, relationship)

                if 'update' in action.lower():
                    # Remove the existing relationship and add the new one
                    graph.remove_edge(similar[0], similar[1])
                    graph.add_edge(relationship.source.id,
                                   relationship.target.id, type=relationship.type)

                elif 'merge' in action.lower():
                    # Add the new relationship, as it complements the existing one
                    graph.add_edge(relationship.source.id,
                                   relationship.target.id, type=relationship.type)
                elif 'discard' in action.lower():
                    # Do nothing, as the new relationship is redundant
                    continue
                elif 'not-connected' in action.lower():
                    # Do nothing, as the relationships are unrelated or refer to different contexts
                    graph.add_edge(relationship.source.id,
                                   relationship.target.id, type=relationship.type)
                else:
                    print("Unknown action:", action)

        # Mark this relationship as processed
        processed_relationships.add(
            (relationship.source.id, relationship.target.id, relationship.type))


def compare_node_similarity(graph, node_id1, node_id2):
    """
    Compares the similarity between two nodes based on their textual representations.
    """
    text1 = create_node_edge_representation(graph, node_id1)
    text2 = create_node_edge_representation(graph, node_id2)
    embedding1 = encode_text(text1)
    embedding2 = encode_text(text2)
    similarity = 1 - cosine(embedding1, embedding2)
    return similarity

# Example usage
if __name__ == "__main__":
    # Load or initialize the graph
    
    G = load_graph()
    clear_graph(G)
    texts = [
        # Case 1: New information updates the existing relationship
        "The Great Wall of China is in China.",
        # This should trigger an 'update' action because Asia is a broader context
        "The Great Wall of China is in Asia.",

        # Case 2: New information complements the existing relationship without contradicting it
        "The Great Wall of China is a historical landmark.",
        # This should trigger a 'merge' action as the new information adds context
        "The Great Wall of China is in China.",

        # Case 3: New information is redundant
        "The Great Wall of China is in China.",
        # This should trigger a 'discard' action because the information is identical
        "The Great Wall of China is in China.",

        # Case 4: Unrelated information
        "The Great Wall of China is in China.",
        # This should trigger a 'not connected' action as they are unrelated
        "The Pacific Ocean is the largest ocean on Earth.",

        # Case 5: Unrelated entities or contexts
        "The Mona Lisa is a famous painting.",
        # This should trigger a 'not connected' action as they refer to different entities
        "The Great Wall of China is in China.",

        # Case 6: Complementary but different types of information
        "The Great Wall of China was built to protect against invasions.",
        # This could trigger a 'merge' action if the information is considered complementary
        "The Great Wall of China is an architectural marvel.",

        # Case 7: New information that refines the existing information
        "The Louvre is a famous museum.",
        # This should trigger a 'merge' action as it refines the existing information about the Louvre
        "The Louvre is in Paris, France.",

        # Case 8: Information that suggests a possible update to existing data
        "The Eiffel Tower was the tallest man-made structure until 1930.",
        # This should trigger an 'update' action with more specific details
        "The Eiffel Tower was the tallest man-made structure until the Chrysler Building was completed in 1930.",

        # Case 9: Information that is related but belongs to a different context
        "The Eiffel Tower is a famous Parisian landmark.",
        # This could be 'not connected' or 'merge' based on context, but typically 'not connected' if no direct relation is mentioned
        "The Eiffel Tower was used in various films.",

        # Case 10: Unrelated and contextually different information
        "Shakespeare wrote 'Romeo and Juliet'.",
        # This should trigger a 'not connected' action as they pertain to completely different contexts
        "The Eiffel Tower is in Paris.",

        # Case 11: Contradictory information
        "The Great Wall of China is in China.",
        # This could be an 'update' if it’s a correction or 'not connected' if considering context relevance
        "The Great Wall of China is in Mongolia.",

        # Case 12: New information that supplements existing data with different attributes
        "The Great Wall of China stretches over 13,000 miles.",
        # This could be 'merge' as it adds supplemental information
        "The Great Wall of China has watchtowers and fortresses.",
    ]



    for text in texts:
        nodes, relationships = process_text_to_graph(text)
        update_graph_with_new_data(G, nodes, relationships)
    save_graph(G)

    
    print(G.nodes(data=True))
    print(G.edges(data=True))

