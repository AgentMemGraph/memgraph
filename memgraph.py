from vectordb import ChromaDBManager
from graphmanager import GraphManager
import uuid
import json
import openai
from openai import OpenAI
client = OpenAI()

class MemgraphMemory:
    def __init__(self, collection_name='all-my-documents', graph_filename='graph2.pkl'):
        self.db_manager = ChromaDBManager(collection_name)
        chroma_client = self.db_manager.client
        self.graph_manager = GraphManager(filename=graph_filename, chroma_client = chroma_client)
        
        # self.graph_manager.clear_graph()

    def add(self, data, user_id=None, metadata=None):
        """
        Add a new memory to both the vector store and the graph.
        """
        # Generate a unique ID
        doc_id = str(uuid.uuid4())

        # Ensure data is a string
        if not isinstance(data, str):
            raise ValueError("Data must be a string.")

        # Add to graph
        nodes, edges = self.graph_manager.add_documents(data)
        print(nodes, edges)

        # Include nodes and edges in the metadata
        metadata = metadata or {}
        nodes_json = json.dumps([node.id for node in nodes])
        edges_json = json.dumps(
            [{'source': edge.source.id, 'target': edge.target.id, 'type': edge.type} for edge in edges])
        metadata.update({
            'nodes': nodes_json,
            'edges': edges_json,
            'user_id': user_id
        })
        print(metadata)

        # Add to vector store
        self.db_manager.add_documents(
            [data], metadatas=[metadata], ids=[doc_id])

        print(f"Memory with ID '{doc_id}' added successfully.")
        return doc_id

    def get_all(self, user_id=None):
        """
        Retrieve all memories from the vector store and graph.
        """
        # Retrieve from vector store
        results = self.db_manager.get_all()
        print(results)
        return results

    def get(self, memory_id):
        """
        Retrieve a single memory by ID from the vector store and graph.
        """
        # Retrieve from vector store
        result = self.db_manager.get_document_by_id(memory_id)

        # Retrieve from graph (if needed)
        node_data = self.graph_manager.create_node_edge_representation(
            memory_id)

        return {
            "document": result,
            "node_data": node_data
        }

    def get_formatted_documents_and_metadatas(self,data):
        documents = data.get('documents', [])
        metadatas = data.get('metadatas', [])

        # Format documents
        formatted_documents = "Documents:\n" + \
            "\n".join(f"- {doc}" for doc in documents[0])

        # Format metadatas
        formatted_metadatas = "Metadatas:\n"
        for metadata in metadatas[0]:
            edges = metadata.get('edges', 'No edges')
            nodes = metadata.get('nodes', 'No nodes')
            user_id = metadata.get('user_id', 'No user_id')

            formatted_metadatas += f"\nUser ID: {user_id}\nEdges: {edges}\nNodes: {nodes}\n"

        return f"{formatted_documents}\n{formatted_metadatas}"
    def search(self, query, user_id=None):
        """
        Search for memories based on a query in the vector store.
        """
        prompt = "You will be given a text. Extract all entites from the text. Return your answer in the following json format: {\"entities\": [\"entity1\", \"entity2\", ...]}\n\nText: " + query
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="gpt-4o-mini",
            response_format={"type": "json_object"}
        )

        ner =  json.loads(response.choices[0].message.content)['entities']
        graph_res = []
        for entity in ner:
            graph_res.append(self.graph_manager.get_node_edges(entity))
        results = self.db_manager.query(
            query_texts=[query], n_results=10)  # Adjust n_results as needed
        formatted_results = self.get_formatted_documents_and_metadatas(results)
        return formatted_results,graph_res

    def update(self, memory_id, data):
        """
        Update an existing memory in both the vector store and the graph.
        """
        # Update vector store
        self.db_manager.update_document(memory_id, data)

        # Update graph
        self.graph_manager.add_documents(data)

        print(f"Memory with ID '{memory_id}' updated successfully.")
        return memory_id

    def delete(self, memory_id):
        """
        Delete a memory by ID from both the vector store and the graph.
        """
        # Delete from vector store
        self.db_manager.delete_document(memory_id)

        # TODO: Implement deletion from graph
        # self.graph_manager.graph.remove_node(memory_id)
        print(f"Memory with ID '{memory_id}' deleted successfully.")

    def delete_all(self, user_id=None):
        """
        Delete all memories from both the vector store and the graph.
        """
        # Delete from vector store
        raise NotImplementedError
        all_ids = self.db_manager.collection.get_all_ids()
        for doc_id in all_ids:
            self.db_manager.delete_document(doc_id)

        # Delete from graph
        self.graph_manager.clear_graph()
        print(f"All memories deleted.")

    def reset(self):
        """
        Reset all memories in both the vector store and the graph.
        """
        # self.delete_all()
        raise NotImplementedError
        self.db_manager.delete_all()
        print("All memories have been reset.")
