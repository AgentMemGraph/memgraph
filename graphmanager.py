import os
import pickle
import networkx as nx
import torch
from scipy.spatial.distance import cosine
from transformers import BertTokenizer, BertModel
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
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


class GraphManager:
    def __init__(self, filename='graph2.pkl'):
        self.filename = filename
        self.graph = self.load_graph()
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4-turbo")
        self.llm_transformer = LLMGraphTransformer(llm=self.llm)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def save_graph(self):
        """
        Saves the graph to a file using pickle.
        """
        with open(self.filename, 'wb') as f:
            pickle.dump(self.graph, f, pickle.HIGHEST_PROTOCOL)

    def load_graph(self):
        """
        Loads the graph from a file using pickle.
        """
        if os.path.exists(self.filename):
            with open(self.filename, 'rb') as f:
                return pickle.load(f)
        else:
            return nx.DiGraph()

    def clear_graph(self):
        """
        Clears all nodes and edges from the graph.
        """
        self.graph.clear()
        print("Graph has been cleared.")

    def add_documents(self, documents):
        """
        Converts documents to graph nodes and relationships.
        """
        nodes, relationships = self.process_text_to_graph(documents)
        added_nodes, added_relationships = self.update_graph_with_new_data(nodes, relationships)
        return added_nodes, added_relationships

    def process_text_to_graph(self, text):
        """
        Converts text to a graph using LLM-based transformation.
        """
        documents = [Document(page_content=text)]
        graph_documents = self.llm_transformer.convert_to_graph_documents(
            documents)
        return graph_documents[0].nodes, graph_documents[0].relationships

    def create_node_edge_representation(self, node_id):
        """
        Creates a textual representation of a node and its edges.
        """
        if node_id not in self.graph.nodes:
            return f"Node {node_id} not found in the graph."
        node_data = self.graph.nodes[node_id]
        descriptions = [f"{node_id}"]
        for target_id in self.graph.neighbors(node_id):
            edge_type = self.graph[node_id][target_id].get('type', 'UNKNOWN')
            descriptions.append(f"{edge_type} {target_id}")
        return ' '.join(descriptions)

    def encode_text(self, text):
        """
        Encodes text into BERT embeddings.
        """
        inputs = self.tokenizer(text, return_tensors='pt',
                                truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    def get_text_representation(self, relationship):
        """
        Creates a detailed textual representation for a relationship.
        """
        source_node = relationship.source.id
        source_type = relationship.source.type
        target_node = relationship.target.id
        target_type = relationship.target.type
        relationship_type = relationship.type

        return (f"Description: This relationship indicates that {source_node}, a {source_type} has a {relationship_type} towards {target_node}, a {target_type}.")

    def get_all_relationship_texts(self):
        """
        Retrieves the text representation for every relationship in the graph.
        """
        relationship_texts = {}
        for source, target, data in self.graph.edges(data=True):
            relationship = Relationship(
                source=Node(id=source, node_type=self.graph.nodes[source].get(
                    'type', 'unknown')),
                target=Node(id=target, node_type=self.graph.nodes[target].get(
                    'type', 'unknown')),
                relationship_type=data.get('type', 'unknown')
            )
            text_representation = self.get_text_representation(relationship)
            relationship_texts[(source, target)] = text_representation
        return relationship_texts

    def find_similar_relationships(self, new_relationship, threshold=0.8):
        """
        Finds existing relationships in the graph that are similar to a new relationship.
        """
        similar_relationships = []
        new_text = self.get_text_representation(new_relationship)
        new_embedding = self.encode_text(new_text)

        for source, target, data in self.graph.edges(data=True):
            source_node_data = self.graph.nodes[source]
            target_node_data = self.graph.nodes[target]

            source_node = Node(
                id=source, node_type=source_node_data.get('type', 'unknown'))
            target_node = Node(
                id=target, node_type=target_node_data.get('type', 'unknown'))

            relationship = Relationship(
                source=source_node, target=target_node, relationship_type=data.get('type', 'unknown'))
            existing_text = self.get_text_representation(relationship)
            existing_embedding = self.encode_text(existing_text)
            similarity = 1 - cosine(new_embedding, existing_embedding)

            if similarity > threshold:
                similar_relationships.append(
                    (source, target, data, similarity))

        return similar_relationships

    def resolve_conflict_with_llm(self, existing_relationship, new_relationship):
        """
        Resolves conflicts between an existing relationship and a new relationship using an LLM.
        """
        existing_text = self.get_text_representation(existing_relationship)
        new_text = self.get_text_representation(new_relationship)
        print(existing_text, new_text)
        prompt = (
            "We have two similar relationships in a knowledge graph. Here are their descriptions:\n\n"
            f"Existing Relationship: {existing_text}\n\n"
            f"New Relationship: {new_text}\n\n"
            "Please determine the appropriate action to take. Choose from one of the following options and provide a brief explanation:\n\n"
            "1. **Update**: If the new relationship provides more recent or accurate information that should replace the existing relationship.\n\n"
            "2. **Merge**: If the new relationship adds valuable information to the existing relationship without contradicting it.\n\n"
            "3. **Discard**: If the new relationship does not add any new value or is redundant.\n\n"
            "4. **Not Connected**: If the existing relationship and the new relationship are unrelated or refer to different entities or contexts.\n\n"
            "Please respond with one of the four options: 'update,' 'merge,' 'discard,' or 'not-connected'. Respond with only one word."
        )

        client = OpenAI()
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="gpt-4o-mini",
        )

        return response.choices[0].message.content


    def update_graph_with_new_data(self, nodes, relationships):
        """
        Updates the graph with new nodes and relationships, removing duplicates.
        Returns the Node and Relationship objects that were actually added.
        """
        added_nodes = []
        added_relationships = []
        processed_relationships = set()

        for node in nodes:
            if not self.graph.has_node(node.id):
                self.graph.add_node(node.id, type=node.type)
                # Create and add the Node object
                added_nodes.append(Node(id=node.id, node_type=node.type))

        for relationship in relationships:
            if (relationship.source.id, relationship.target.id, relationship.type) in processed_relationships:
                continue

            similar_relationships = self.find_similar_relationships(relationship)

            if not similar_relationships:
                self.graph.add_edge(
                    relationship.source.id, relationship.target.id, type=relationship.type)
                # Create and add the Relationship object
                added_relationships.append(Relationship(
                    source=Node(id=relationship.source.id, node_type=self.graph.nodes[relationship.source.id].get(
                        'type', 'unknown')),
                    target=Node(id=relationship.target.id, node_type=self.graph.nodes[relationship.target.id].get(
                        'type', 'unknown')),
                    relationship_type=relationship.type
                ))
            else:
                for similar in similar_relationships:
                    existing_relationship = Relationship(
                        source=Node(id=similar[0], node_type=self.graph.nodes[similar[0]].get(
                            'type', 'unknown')),
                        target=Node(id=similar[1], node_type=self.graph.nodes[similar[1]].get(
                            'type', 'unknown')),
                        relationship_type=similar[2].get('type', 'unknown')
                    )
                    action = self.resolve_conflict_with_llm(
                        existing_relationship, relationship)
                    print(existing_relationship, relationship)
                    print()
                    print('action:', action)
                    print()
                    if 'update' in action.lower():
                        self.graph.remove_edge(similar[0], similar[1])
                        self.graph.add_edge(
                            relationship.source.id, relationship.target.id, type=relationship.type)
                        added_relationships.append(Relationship(
                            source=Node(id=relationship.source.id, node_type=self.graph.nodes[relationship.source.id].get(
                                'type', 'unknown')),
                            target=Node(id=relationship.target.id, node_type=self.graph.nodes[relationship.target.id].get(
                                'type', 'unknown')),
                            relationship_type=relationship.type
                        ))

                    elif 'merge' in action.lower():
                        self.graph.add_edge(
                            relationship.source.id, relationship.target.id, type=relationship.type)
                        added_relationships.append(Relationship(
                            source=Node(id=relationship.source.id, node_type=self.graph.nodes[relationship.source.id].get(
                                'type', 'unknown')),
                            target=Node(id=relationship.target.id, node_type=self.graph.nodes[relationship.target.id].get(
                                'type', 'unknown')),
                            relationship_type=relationship.type
                        ))

                    elif 'discard' in action.lower():
                        continue

                    elif 'not-connected' in action.lower():
                        self.graph.add_edge(
                            relationship.source.id, relationship.target.id, type=relationship.type)
                        added_relationships.append(Relationship(
                            source=Node(id=relationship.source.id, node_type=self.graph.nodes[relationship.source.id].get(
                                'type', 'unknown')),
                            target=Node(id=relationship.target.id, node_type=self.graph.nodes[relationship.target.id].get(
                                'type', 'unknown')),
                            relationship_type=relationship.type
                        ))
                    else:
                        print("Unknown action:", action)

            processed_relationships.add(
                (relationship.source.id, relationship.target.id, relationship.type))

        return added_nodes, added_relationships





# Example usage:
if __name__ == "__main__":
    manager = GraphManager()

    # Add documents to the graph
    documents = [
        "The Great Wall of China is in China.",
        "The Great Wall of China is in Asia.",
        "The Great Wall of China is a historical landmark.",
        "The Great Wall of China is in China.",
        "The Great Wall of China is in China.",
        "The Pacific Ocean is the largest ocean on Earth.",
        "The Mona Lisa is a famous painting.",
        "The Great Wall of China is in China.",
        "The Great Wall of China was built to protect against invasions.",
        "The Great Wall of China is an architectural marvel.",
        "The Louvre is a famous museum.",
        "The Louvre is in Paris, France.",
        "The Eiffel Tower was the tallest man-made structure until 1930.",
        "The Eiffel Tower was the tallest man-made structure until the Chrysler Building was completed in 1930.",
        "The Eiffel Tower is a famous Parisian landmark.",
        "The Eiffel Tower was used in various films."
    ]

    for document in documents:# Add the documents to the graph
     manager.add_documents(document)
# Add the documents to the do
    # manager.add_documents(documents)

    # Compare node similarity
    # similarity = manager.compare_node_similarity('node1', 'node2')
    

    # Save the graph
    manager.save_graph()
    print(manager.get_all_relationship_texts())
