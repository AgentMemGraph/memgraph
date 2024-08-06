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
import chromadb
from chromadb.config import Settings
import hashlib
import json
from create_kg import create_graph

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
    def __init__(self, filename='graph.pkl', embedding_db_path='./graph_embedding_db', chroma_client=None):
        self.filename = filename
        self.graph = self.load_graph()
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4-turbo")
        self.llm_transformer = LLMGraphTransformer(llm=self.llm)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        # self.chroma_client = chroma_client if chroma_client else chromadb.Client(
        #     Settings(persist_directory=embedding_db_path))
        self.chroma_client = chromadb.PersistentClient(
            path=embedding_db_path)

        self.node_collection = self.chroma_client.get_or_create_collection(
            "node_embeddings")
        self.relationship_collection = self.chroma_client.get_or_create_collection(
            "relationship_embeddings")
        # collection = self.chroma_client.get_collection(
        #     name="relationship_embeddings")
        # print(self.relationship_collection.get())

    def save_graph(self):
        """
        Saves the graph to a file using pickle.
        """
        with open(self.filename, 'wb') as f:
            pickle.dump(self.graph, f, pickle.HIGHEST_PROTOCOL)
        # self.chroma_client.persist()

    def load_graph(self):
        """
        Loads the graph from a file using pickle.
        """
        if os.path.exists(self.filename):
            with open(self.filename, 'rb') as f:
                return pickle.load(f)
        else:
            # return nx.MultiDiGraph()
            return nx.MultiDiGraph()

    def clear_graph(self):
        """
        Clears all nodes and edges from the graph.
        """
        self.graph.clear()
        self.chroma_client.delete_collection('node_embeddings')
        self.chroma_client.delete_collection('relationship_embeddings')
        self.node_collection = self.chroma_client.get_or_create_collection(
            "node_embeddings")
        self.relationship_collection = self.chroma_client.get_or_create_collection(
            "relationship_embeddings")
        print("Graph has been cleared.")

    def add_documents(self, documents, update = True):
        """
        Converts documents to graph nodes and relationships.
        """
        # self.graph = self.load_graph()

        nodes, relationships = self.process_text_to_graph(documents)
        if not update: 
            added_nodes, added_relationships = self.add_without_updating(
                nodes, relationships)
        else: #Default Case: Will Update
            added_nodes, added_relationships = self.update_graph_with_new_data(
                nodes, relationships)
        self.save_graph()

        return added_nodes, added_relationships

    def add_without_updating(self, nodes, relationships):
        """
        Adds nodes and relationships to the graph without updating or resolving conflicts.
        """
        for node in nodes:
            if not self.graph.has_node(node.id):
                self.graph.add_node(node.id, type=node.type)
                self.add_node_embedding(node.id, node.type)

        for relationship in relationships:
            if not self.graph.has_edge(relationship.source.id, relationship.target.id):
                self.graph.add_edge(
                    relationship.source.id, relationship.target.id, type=relationship.type)
                self.add_relationship_embedding(
                    relationship.source.id, relationship.target.id, relationship.type)

        return nodes, relationships

    def process_text_to_graph(self, text):
        """
        Converts text to a graph using LLM-based transformation.
        """
        # documents = [Document(page_content=text)]
        # graph_documents = self.llm_transformer.convert_to_graph_documents(
        #     documents)
        # return graph_documents[0].nodes, graph_documents[0].relationships
        nodes, relationships = create_graph(text)
        return nodes, relationships
        

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

    def add_node_embedding(self, node_id, node_type):
        """
        Adds a node embedding to ChromaDB.
        """
        text_representation = f"{node_id} ({node_type})"
        embedding = self.encode_text(text_representation)
        self.node_collection.add(
            embeddings=[embedding.tolist()],
            documents=[text_representation],
            ids=[node_id]
        )

    def add_relationship_embedding(self, source_id, target_id, relationship_type):
        """
        Adds a relationship embedding to ChromaDB.
        """
        text_representation = self.get_text_representation(Relationship(
            source=Node(id=source_id, node_type=self.graph.nodes[source_id].get(
                'type', 'unknown')),
            target=Node(id=target_id, node_type=self.graph.nodes[target_id].get(
                'type', 'unknown')),
            relationship_type=relationship_type
        ))
        embedding = self.encode_text(text_representation)
        # print(embedding.tolist())
        embedding = embedding.tolist()
        relationship_id = f"{source_id}|{target_id}|{relationship_type}"
        x = self.relationship_collection.add(
            embeddings=[embedding],
            documents=[text_representation],
            ids=[relationship_id]
        )

    def print_chroma_embeddings(self):
        """
        Prints the embeddings stored in ChromaDB with clear and formatted output.
        """
        # Print Node Embeddings
        print("===== Node Embeddings =====")
        node_embeddings = self.node_collection.get()
        if 'documents' in node_embeddings:
            for i, node in enumerate(node_embeddings['documents'], start=1):
                print(f"  Node {i}: {node}")
        else:
            print("No node embeddings found.")

        # Print Relationship Embeddings
        print("\n===== Relationship Embeddings =====")
        relationship_embeddings = self.relationship_collection.get()
        if 'documents' in relationship_embeddings:
            for i, relationship in enumerate(relationship_embeddings['documents'], start=1):
                print(f"  Relationship {i}: {relationship}")
        else:
            print("No relationship embeddings found.")

    def delete_relationship_embedding(self, source_id, target_id, relationship_type):
        """
        Deletes a relationship embedding from ChromaDB.
        """

        relationship_id = f"{source_id}_{target_id}_{relationship_type}"

        try:
            # Remove the relationship embedding from ChromaDB
            self.relationship_collection.delete(ids=[relationship_id])

        except Exception as e:
            print(
                f'Failed to delete relationship embedding for ID: {relationship_id}. Error: {str(e)}')

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

    def get_all_relationships(self):
        """
        Retrieves all relationships in the graph.
        """
        relationships = []
        for source, target, data in self.graph.edges(data=True):
            relationships.append((
                source,
                target,
                {'type': data.get('type', 'unknown')}
            ))
        return relationships

    # def find_similar_relationships(self, new_relationship, threshold=0.8):
    #     """
    #     Finds existing relationships in the graph that are similar to a new relationship.
    #     """
    #     similar_relationships = []
    #     new_text = self.get_text_representation(new_relationship)
    #     new_embedding = self.encode_text(new_text)

    #     for source, target, data in self.graph.edges(data=True):
    #         source_node_data = self.graph.nodes[source]
    #         target_node_data = self.graph.nodes[target]

    #         source_node = Node(
    #             id=source, node_type=source_node_data.get('type', 'unknown'))
    #         target_node = Node(
    #             id=target, node_type=target_node_data.get('type', 'unknown'))

    #         relationship = Relationship(
    #             source=source_node, target=target_node, relationship_type=data.get('type', 'unknown'))
    #         existing_text = self.get_text_representation(relationship)
    #         existing_embedding = self.encode_text(existing_text)
    #         similarity = 1 - cosine(new_embedding, existing_embedding)

    #         if similarity > threshold:
    #             similar_relationships.append(
    #                 (source, target, data, similarity))

    #     return similar_relationships

    def find_similar_relationships(self, new_relationship, threshold=0.8):
        """
        Finds existing relationships in the graph that are similar to a new relationship.
        """
        new_text = self.get_text_representation(new_relationship)
        new_embedding = self.encode_text(new_text)

        results = self.relationship_collection.query(
            query_embeddings=[new_embedding.tolist()],
            n_results=10,
            include=['embeddings']
        )

        # print(results)
        # print(results)
        # print(self.relationship_collection.get())
        similar_relationships = []
        for i, relationship_id in enumerate(results['ids'][0]):
            similarity = 1 - cosine(new_embedding,
                                    np.array(results['embeddings'][0][i]))
            if similarity > threshold:
                try:
                    source_id, target_id, relationship_type = relationship_id.split(
                        '|')
                    similar_relationships.append((
                        source_id,
                        target_id,
                        {'type': relationship_type},
                        similarity
                    ))
                except Exception as e:
                    print(e, relationship_id)

        return similar_relationships

    def resolve_conflict_with_llm(self, existing_relationships, new_relationship):
        """
        Resolves conflicts between an existing relationship and a new relationship using an LLM.
        """
        existing_relationship_texts = [self.get_text_representation(
            existing_relationship) for existing_relationship in existing_relationships]
        new_text = self.get_text_representation(new_relationship)
        # print(existing_text, new_text)
        # prompt = (
        #     "We have two similar relationships in a knowledge graph. Here are their descriptions:\n\n"
        #     f"Existing Relationship: {existing_text}\n\n"
        #     f"New Relationship: {new_text}\n\n"
        #     "Please determine the appropriate action to take. Choose from one of the following options and provide a brief explanation:\n\n"
        #     "1. **Update**: If the new relationship provides more recent or accurate information that should replace the existing relationship.\n\n"
        #     "2. **Merge**: If the new relationship adds valuable information to the existing relationship without contradicting it.\n\n"
        #     "3. **Discard**: If the new relationship does not add any new value or is redundant.\n\n"
        #     "4. **Not Connected**: If the existing relationship and the new relationship are unrelated or refer to different entities or contexts.\n\n"
        #     "Please respond with one of the four options: 'update,' 'merge,' 'discard,' or 'not-connected'. Respond with only one word."
        # )
        # prompt = ("We have two relationships in a knowledge graph. Here are their descriptions:\n\n"
        #           f"Existing Relationship: {existing_text}\n\n"
        #           f"New Relationship: {new_text}\n\n"
        #           "Please determine the appropriate action to take. Choose from one of the following options and provide a brief explanation:\n\n"
        #           "1. **Retain Existing**: If the existing relationship is identical or similar to the new one, and the new relationship should not be added.\n\n"
        #           "2. **Add Both**: If the new relationship complements the existing relationship, and both should be retained.\n\n"
        #           "3. **Replace Existing**: If the new relationship conflicts with the existing relationship, and the existing relationship should be removed and replaced with the new one.\n\n"
        #           "4. **Not Connected**: If the existing relationship and the new relationship are unrelated or refer to different entities or contexts.\n\n"
        #           "Please respond with one of the four options: 'retain-existing,' 'add-both,' 'replace-existing,' or 'not-connected'. Respond with only one word.")

        prompt = ("We have a list of relationships in a knowledge graph, and a new relationship to be added. Here are their descriptions:\n\n"
                  f"Existing Relationship: {existing_relationship_texts}\n\n"
                  f"New Relationship: {new_text}\n\n"
                  "Please determine the appropriate action to take for each existing relationship Choose from one of the following options and provide a brief explanation:\n\n"
                  "1. **Retain Existing**: If the existing relationship is identical or similar to the new one, and the new relationship should not be added.\n\n"
                  "2. **Add Both**: If the new relationship complements the existing relationship, and both should be retained.\n\n"
                  "3. **Replace Existing**: If the new relationship conflicts with the existing relationship, and the existing relationship should be removed and replaced with the new one.\n\n"
                  "4. **Not Connected**: If the existing relationship and the new relationship are unrelated or refer to different entities or contexts.\n\n"
                  "5. You must respond with one of the four actions for each existing relationship: 'retain-existing,' 'add-both,' 'replace-existing,' or 'not-connected'. Respond with only one word."
                  "You must respond in a JSON format as follows: {'response': [{'relationship':'existing_relationship','action': 'action'}, {'relationship':'existing_relationship','action': 'action'}, ...]}")
        client = OpenAI()
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="gpt-4o-mini",
            response_format={"type": "json_object"}
        )
        # print(prompt, response.choices[0].message.content)
        return json.loads(response.choices[0].message.content)

    # def update_graph_with_new_data(self, nodes, relationships):
        """
        Updates the graph with new nodes and relationships, removing duplicates.
        Returns the Node and Relationship objects that were actually added.
        """
        # added_nodes = []
        # added_relationships = []
        # processed_relationships = set()

        # for node in nodes:
        #     if not self.graph.has_node(node.id):
        #         self.graph.add_node(node.id, type=node.type)
        #         self.add_node_embedding(node.id, node.type)
        #         # Create and add the Node object
        #         added_nodes.append(Node(id=node.id, node_type=node.type))

        # for relationship in relationships:
        #     if (relationship.source.id, relationship.target.id, relationship.type) in processed_relationships:
        #         continue

        #     similar_relationships = self.find_similar_relationships(relationship)

        #     if not similar_relationships:
        #         self.graph.add_edge(
        #             relationship.source.id, relationship.target.id, type=relationship.type)
        #         self.add_relationship_embedding(
        #             relationship.source.id, relationship.target.id, relationship.type)
        #         # Create and add the Relationship object
        #         added_relationships.append(Relationship(
        #             source=Node(id=relationship.source.id, node_type=self.graph.nodes[relationship.source.id].get(
        #                 'type', 'unknown')),
        #             target=Node(id=relationship.target.id, node_type=self.graph.nodes[relationship.target.id].get(
        #                 'type', 'unknown')),
        #             relationship_type=relationship.type
        #         ))
        #     else:
        #         for similar in similar_relationships:
        #             existing_relationship = Relationship(
        #                 source=Node(id=similar[0], node_type=self.graph.nodes[similar[0]].get(
        #                     'type', 'unknown')),
        #                 target=Node(id=similar[1], node_type=self.graph.nodes[similar[1]].get(
        #                     'type', 'unknown')),
        #                 relationship_type=similar[2].get('type', 'unknown')
        #             )
        #             action = self.resolve_conflict_with_llm(
        #                 existing_relationship, relationship)

        #             print(existing_relationship, relationship)
        #             print()
        #             print('action:', action)
        #             print()
        #             if 'update' in action.lower():
        #                 self.graph.remove_edge(similar[0], similar[1])
        #                 self.delete_relationship_embedding(
        #                     similar[0], similar[1], similar[2].get('type', 'unknown'))  # Add this line
        #                 self.graph.add_edge(relationship.source.id,
        #                                     relationship.target.id, type=relationship.type)
        #                 self.add_relationship_embedding(
        #                     relationship.source.id, relationship.target.id, relationship.type)
        #                 added_relationships.append(Relationship(
        #                     source=Node(id=relationship.source.id, node_type=self.graph.nodes[relationship.source.id].get(
        #                         'type', 'unknown')),
        #                     target=Node(id=relationship.target.id, node_type=self.graph.nodes[relationship.target.id].get(
        #                         'type', 'unknown')),
        #                     relationship_type=relationship.type
        #                 ))

        #             elif 'merge' in action.lower():
        #                 self.graph.add_edge(
        #                     relationship.source.id, relationship.target.id, type=relationship.type)
        #                 self.add_relationship_embedding(
        #                     relationship.source.id, relationship.target.id, relationship.type)
        #                 added_relationships.append(Relationship(
        #                     source=Node(id=relationship.source.id, node_type=self.graph.nodes[relationship.source.id].get(
        #                         'type', 'unknown')),
        #                     target=Node(id=relationship.target.id, node_type=self.graph.nodes[relationship.target.id].get(
        #                         'type', 'unknown')),
        #                     relationship_type=relationship.type
        #                 ))

        #             elif 'discard' in action.lower():
        #                 continue

        #             elif 'not-connected' in action.lower():
        #                 self.graph.add_edge(
        #                     relationship.source.id, relationship.target.id, type=relationship.type)
        #                 self.add_relationship_embedding(
        #                     relationship.source.id, relationship.target.id, relationship.type)
        #                 added_relationships.append(Relationship(
        #                     source=Node(id=relationship.source.id, node_type=self.graph.nodes[relationship.source.id].get(
        #                         'type', 'unknown')),
        #                     target=Node(id=relationship.target.id, node_type=self.graph.nodes[relationship.target.id].get(
        #                         'type', 'unknown')),
        #                     relationship_type=relationship.type
        #                 ))
        #             else:
        #                 print("Unknown action:", action)

        #     processed_relationships.add(
        #         (relationship.source.id, relationship.target.id, relationship.type))

        # return added_nodes, added_relationships

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
                self.add_node_embedding(node.id, node.type)
                added_nodes.append(Node(id=node.id, node_type=node.type))
            else:
                self.graph.nodes[node.id]['type'] = node.type

        for relationship in relationships:
            if (relationship.source.id, relationship.target.id, relationship.type) in processed_relationships:
                continue

            similar_relationships = self.find_similar_relationships(
                relationship)
            should_add = True
            if similar_relationships:
                existing_relationships = [Relationship(
                    source=Node(id=similar[0], node_type=self.graph.nodes[similar[0]].get(
                        'type', 'unknown')),
                    target=Node(id=similar[1], node_type=self.graph.nodes[similar[1]].get(
                        'type', 'unknown')),
                    relationship_type=similar[2].get('type', 'unknown')
                ) for similar in similar_relationships]
                actions = self.resolve_conflict_with_llm(
                    existing_relationships, relationship)

                if len(actions['response']) != len(similar_relationships):
                    # Retry once
                    actions = self.resolve_conflict_with_llm(
                        existing_relationships, relationship)

                    if len(actions['response']) != len(similar_relationships):
                        raise ValueError(
                            "Mismatch between the number of responses and similar relationships")

                for i in range(len(actions['response'])):
                    action = actions['response'][i].get('action', 'unknown')
                    similar = similar_relationships[i]
                    if 'replace-existing' in action.lower() or 'retain-existing' in action.lower():
                        # Remove the existing relationship
                        try:
                            # pass
                            self.graph.remove_edge(similar[0], similar[1])
                            self.delete_relationship_embedding(
                                similar[0], similar[1], similar[2].get('type', 'unknown'))
                        except Exception as e:
                            # print("SIMILAR ", similar)
                            # print("Exception line 508")
                            # rels = self.get_all_relationships()
                            # for rel in rels:
                            #     print(rel)
                            print(e)
                        continue  # Exit the loop to add the new relationship

                    elif 'add-both' in action.lower():
                        # Add both relationships without removing the existing one
                        continue  # Exit the loop to add the new relationship

                    elif 'not-connected' in action.lower():
                        continue  # Check the next similar relationship

                    else:
                        print("Unknown action:", action)
                        # should_add = False
                        break  # Exit the loop and don't add the new relationship

                # for similar in similar_relationships:
                #     try:
                #         existing_relationship = Relationship(
                #             source=Node(id=similar[0], node_type=self.graph.nodes[similar[0]].get(
                #                 'type', 'unknown')),
                #             target=Node(id=similar[1], node_type=self.graph.nodes[similar[1]].get(
                #                 'type', 'unknown')),
                #             relationship_type=similar[2].get('type', 'unknown')
                #         )
                #     except Exception as e:
                #         existing_relationship = Relationship(
                #             source=Node(id=similar[0], node_type='unknown'),
                #             target=Node(id=similar[1],node_type='unknown'),
                #             relationship_type=similar[2].get('type', 'unknown')
                #         )
                #     action = self.resolve_conflict_with_llm(
                #         [existing_relationship], relationship)
                #     # print("A: ",similar, "B: :",relationship)
                #     # print()
                #     # print('action:', action)
                #     # print()

                #     if 'replace-existing' in action.lower() or 'retain-existing' in action.lower():
                #         # Remove the existing relationship
                #         try:
                #             # pass
                #             self.graph.remove_edge(similar[0], similar[1])
                #             self.delete_relationship_embedding(
                #                 similar[0], similar[1], similar[2].get('type', 'unknown'))
                #         except Exception as e:
                #             # print("SIMILAR ", similar)
                #             # print("Exception line 508")
                #             # rels = self.get_all_relationships()
                #             # for rel in rels:
                #             #     print(rel)
                #             print(e)
                #         continue  # Exit the loop to add the new relationship

                #     elif 'add-both' in action.lower():
                #         # Add both relationships without removing the existing one
                #         continue  # Exit the loop to add the new relationship

                #     elif 'not-connected' in action.lower():
                #         continue  # Check the next similar relationship

                #     else:
                #         print("Unknown action:", action)
                #         # should_add = False
                #         break  # Exit the loop and don't add the new relationship

            # print("Adding relationship", relationship)
            # print("ALL GRAPH NODES ", list(self.graph.nodes))
            # print(relationship.source.id,
            #       relationship.target.id, relationship.type)
            self.graph.add_edge(relationship.source.id,
                                relationship.target.id, type=relationship.type)
            # print(self.graph.edges(data = True))
            self.add_relationship_embedding(
                relationship.source.id, relationship.target.id, relationship.type)
            added_relationships.append(Relationship(
                source=Node(id=relationship.source.id, node_type=self.graph.nodes[relationship.source.id].get(
                    'type', 'unknown')),
                target=Node(id=relationship.target.id, node_type=self.graph.nodes[relationship.target.id].get(
                    'type', 'unknown')),
                relationship_type=relationship.type
            ))

            processed_relationships.add(
                (relationship.source.id, relationship.target.id, relationship.type))

        return added_nodes, added_relationships

    def get_node_edges(self, node_id):
        """
        Retrieves all edges where the given node is either a source or a target.
        """
        if node_id not in self.graph.nodes:
            return []

        # Outgoing edges where the node is a source
        outgoing_edges = [(node_id, target, data)
                          for target, data in self.graph[node_id].items()]

        # Incoming edges where the node is a target
        incoming_edges = [(source, node_id, data)
                          for source, data in self.graph.pred[node_id].items()]

        # Combine both outgoing and incoming edges
        all_edges = outgoing_edges + incoming_edges

        return all_edges

    def get_edge_between_nodes(self, source_node, target_node):
        """
        Returns the edge data if the two nodes are connected, or None if they aren't.

        :param source_node: The ID of the source node
        :param target_node: The ID of the target node
        :return: A Relationship object if the nodes are connected, or None if they aren't
        """
        if self.graph.has_edge(source_node, target_node):
            edge_data = self.graph.get_edge_data(source_node, target_node)
            return (source_node, target_node, {'type': edge_data.get('type', 'unknown')})
        else:
            return None


# Example usage:
if __name__ == "__main__":
    manager = GraphManager(filename='graph20.pkl',
                           embedding_db_path='./graph_embedding_db20')

    texts = [
        "John went to the park today and met his old friend, Sarah.",
        "In the afternoon, John attended a meeting with his boss about the upcoming project.",
        "John wrote in his diary about feeling anxious before his presentation tomorrow.",
        "Sarah told John about her plans to travel to Europe next summer.",
        "John decided to start a new hobby—painting landscapes.",
        "After the meeting, John had dinner with his colleague, James.",
        "John's sister called to inform him about the family reunion next month.",
        "John spent the evening reading a book on mindfulness.",
        "John helped his neighbor fix a flat tire on their car.",
        "John received an invitation to his high school reunion.",
        "John and Sarah went hiking over the weekend.",
        "John felt relieved after finishing his project ahead of the deadline.",
        "John's dog, Max, was not feeling well, so he took him to the vet.",
        "John attended a webinar on digital marketing strategies.",
        "John ran into an old classmate at the grocery store.",
        "John spent the weekend at a cabin in the mountains.",
        "John's boss praised him for his hard work on the recent project.",
        "John bought a new set of paints to explore his artistic side.",
        "John and James discussed their favorite books over lunch.",
        "John spent the afternoon organizing his workspace.",
        "John visited a local art gallery to find inspiration for his paintings.",
        "John's parents sent him a care package with his favorite snacks.",
        "John had a long conversation with Sarah about their future plans.",
        "John was thrilled to receive a promotion at work.",
        "John took Max for a walk in the nearby nature reserve.",
        "John tried a new recipe for dinner and was happy with the results.",
        "John attended a yoga class to help reduce stress.",
        "John and Sarah watched a documentary on wildlife conservation.",
        "John volunteered at a local charity event over the weekend.",
        "John was invited to speak at a conference about his work.",
        "John bought tickets to a concert happening next month.",
        "John's sister visited him for the weekend, and they spent time reminiscing.",
        "John and his friends planned a weekend getaway to the beach.",
        "John was nervous about an upcoming presentation at work.",
        "John took a day off to spend time painting in his studio.",
        "John attended a networking event and made several new connections.",
        "John had a productive day and completed all his tasks ahead of time.",
        "John visited a new coffee shop that opened in his neighborhood.",
        "John decided to start journaling every night before bed.",
        "John and Sarah attended a wedding of a mutual friend.",
        "John spent the afternoon researching new techniques for his paintings.",
        "John was delighted to receive a surprise gift from Sarah.",
        "John and James started working on a joint project at work.",
        "John visited his parents for a family dinner.",
        "John spent the evening catching up on his favorite TV series.",
        "John's sister asked him to be the godfather to her newborn son.",
        "John attended a photography workshop to improve his skills.",
        "John took a long walk along the beach to clear his mind.",
        "John and Sarah decided to plan a trip together next year.",
        "John received positive feedback on his recent work from his team."
    ]

    # /manager.add_documents("John's Team liked John's recent work")

    for document in texts:
        manager.add_documents(document)
        print()
        print("================== NEW DOCUMENT ==================")
        print()
        print()

    # Compare node similarity
    # similarity = manager.compare_node_similarity('node1', 'node2')
    # Save the graph
    manager.save_graph()

    rels = (manager.get_all_relationships())
    for rel in rels:
        print(rel)
    # x = (manager.get_node_edges('John'))
    # for j in x:
    #     print(j)
    # print(manager.get_edge_between_nodes('John', 'Sarah'))
