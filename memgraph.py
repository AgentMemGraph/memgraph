from vectordb import ChromaDBManager
from graphmanager import GraphManager
import uuid
import json
import openai
from openai import OpenAI
client = OpenAI()


class MemgraphMemory:
    def __init__(self, collection_name='all-my-documents', graph_filename='graph.pkl'):
        self.db_manager = ChromaDBManager(collection_name)
        chroma_client = self.db_manager.client
        self.graph_manager = GraphManager(
            filename=graph_filename, chroma_client=chroma_client)

        # self.graph_manager.clear_graph()

    def add(self, data, user_id=None, metadata=None, update=False):
        """
        Add a new memory to both the vector store and the graph.
        """
        # Generate a unique ID
        doc_id = str(uuid.uuid4())

        # Ensure data is a string
        if not isinstance(data, str):
            raise ValueError("Data must be a string.")

        # Add to graph
        nodes, edges = self.graph_manager.add_documents(data, update)

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

        # Add to vector store
        self.db_manager.add_documents(
            [data], metadatas=[metadata], ids=[doc_id])

        # print(f"Memory with ID '{doc_id}' added successfully.")
        return doc_id

    def get_all(self, user_id=None):
        """
        Retrieve all memories from the vector store and graph.
        """
        # Retrieve from vector store
        results = self.db_manager.get_all()

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

    def get_formatted_documents_and_metadatas(self, data, graph_res):
        documents = data.get('documents', [])
        metadatas = data.get('metadatas', [])

        # Format documents
        formatted_documents = "Documents:\n" + \
            "\n".join(f"- {doc}" for doc in documents[0])

        formatted_documents = formatted_documents + "\n\nGraph Results:\n" + \
            "\n".join(f"- {doc}" for doc in graph_res)
        # Format metadatas
        # formatted_metadatas = "Metadatas:\n"
        # for metadata in metadatas[0]:
        #     edges = metadata.get('edges', 'No edges')
        #     nodes = metadata.get('nodes', 'No nodes')
        #     user_id = metadata.get('user_id', 'No user_id')

        #     formatted_metadatas += f"\nUser ID: {user_id}\nEdges: {edges}\nNodes: {nodes}\n"
        return f"{formatted_documents}"

    def search(self, query, user_id=None, max_hops=4):
        """
        Search for memories based on a query in the vector store.
        """
        
        return_result = ""
        all_nodes = list(self.graph_manager.graph.nodes)
        prompt = "You will be given a text. Extract all entites from the text. Each entity should be atomic. E.g. an entity should be a Person, a Location, so on. Here is a list of all the allowed entites you can extract:" + \
            str(all_nodes) + \
            "Return your answer in the following json format: {\"entities\": [\"entity1\", \"entity2\", ...]}\n\nText: " + query
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="gpt-4o-mini",
            response_format={"type": "json_object"}
        )

        ner = json.loads(response.choices[0].message.content)['entities']
        graph_res = []
        for entity in ner:
            graph_res.append(self.graph_manager.get_node_edges(entity))

        results = self.db_manager.query(
            query_texts=[query], n_results=10)  # Adjust n_results as needed
        formatted_results = self.get_formatted_documents_and_metadatas(
            results, graph_res)

        second_prompt = "You have been given a query and a set of related memories. Answer the query using the information provided in the related memories. Do you want further information from the Knowledge Graph? Response must be in JSON format as follows: If no: {\"response\": \"no\"} If yes: {\"response\": \"yes\", \"entities\": [\"entity1\", \"entity2\", ...]}\n\nQuery: " + query + "\n\nRelated Memories: " + formatted_results
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": second_prompt}],
            model="gpt-4o-mini",
            response_format={"type": "json_object"}
        )
        response = json.loads(response.choices[0].message.content)
        if response['response'] == 'yes':
            # print("MULTIHOP ACTIVATED")
            # print(response['entities'])
            for entity in response['entities']:
                graph_res.append(self.graph_manager.get_node_edges(entity))
            formatted_results = self.get_formatted_documents_and_metadatas(
                results, graph_res)
        all_nodes = list(self.graph_manager.graph.nodes)
        graph_res = []
        used_entities = set()
        # Entity extraction prompt
        entity_extraction_prompt = f"""
            Analyze the following text and extract all relevant entities. Each entity should be specific and atomic, such as a Person, Location, Organization, Event, Concept, or Object.
            Allowed entities: {all_nodes}
            Previously used entities: {list(used_entities)}
            Guidelines:
            1. Focus on extracting entities that are directly mentioned or strongly implied in the text.
            2. Avoid overly broad or generic terms.
            3. Include only entities that are likely to be useful for answering the query.
            4. If an entity is not in the allowed list but is crucial to the query, include it anyway.
            5. Prioritize entities that haven't been used before.
            Text: {query}
            Provide your answer in the following JSON format:
            {{
                "entities": ["entity1", "entity2", ...]
            }}
            """

        response = client.chat.completions.create(
            messages=[
                {"role": "user", "content": entity_extraction_prompt}],
            model="gpt-4o-mini",
            response_format={"type": "json_object"}
        )

        ner = json.loads(response.choices[0].message.content)['entities']
        # print('Extracted Entities:', ner)

        new_entities = [
            entity for entity in ner if entity not in used_entities]
        used_entities.update(new_entities)

        for entity in new_entities:
            if entity in self.graph_manager.graph.nodes:
                graph_res.append(
                    entity + ": " + str(self.graph_manager.graph.nodes[entity]))
                graph_res.append(self.graph_manager.get_node_edges(entity))

        formatted_results = "\n\nGraph Results:\n" + \
            "\n".join(f"- {doc}" for doc in graph_res)
        # print(f"AFTER INITIAL QUERY: {formatted_results}")
        analysis_prompt = f"""
            You are an AI assistant tasked with determining if more information is needed to answer a query based on provided information from a Knowledge Graph.
            The primary question is, "Do you have enough information to answer the query based on the provided information?"
            Query: {query}
            Current Information: {formatted_results}
            Allowed Entities: {all_nodes}
            Previously Used Entities: {list(used_entities)}
            Instructions:
            1. Carefully analyze the query and the provided information.
            2. Determine if you have enough information to answer the query or if more hops are needed (and allowed).
            3. If you need additional information and more hops are allowed, specify which entities you need more details about.
            4. Only include entities that are in the Allowed Entities list and not in the Previously Used Entities list.
            5. If the current Information does not have enough information to answer the Query, you should request more information.
            6. Do not use your own knowledge to try and answer the question. Ensure that if you answer 'no', the Current Information is enough to answer the Query. 
            7. You should respond with your reasoning for your answer. This reasoning should be based on why you claim the current information can answer the query. 
            8. You should also create a sub-query, a query which must be answered next, or a simplified version of the current query given the current information.
            Respond in JSON format as follows:
            9. You may not output entities that are previously used. Hence, if you believe that the information may not be present, you may answer 'yes', i.e. if you believe all relevant entities have been accessed. 
            {{
                "response": "yes" or "no",
                "entities": ["entity1", "entity2", ...] (only if more information is needed, otherwise an empty list),
                "reasoning": "Reasoning for your response
                "sub-query": "Sub-query to be answered
            }}
        """
        messages = []
        for hop in range(max_hops):
            # Analysis prompt
            messages.append({"role": "user", "content": analysis_prompt})
            response = client.chat.completions.create(
                messages=messages,
                model="gpt-4o-mini",
                response_format={"type": "json_object"}
            )
            messages.append(
                {"role": "assistant", "content": response.choices[0].message.content})
            # messages.append
            response_data = json.loads(response.choices[0].message.content)
            # print(response_data)
            if response_data['response'] == 'yes' or hop == max_hops - 1:
                # print(f"Multi-hop search completed after {hop + 1} hops.")

                # return formatted_results, 'no', [], response_data['reasoning']
                return formatted_results

            # print(f"HOP {hop + 1} COMPLETED - MORE INFORMATION NEEDED")
            # print("Additional entities requested:", response_data['entities'])
            graph_res = []
            for entity in response_data['entities']:
                try:
                    graph_res.append(self.graph_manager.graph.nodes[entity])
                except Exception as e:
                    pass
                graph_res.append(self.graph_manager.get_node_edges(entity))
            formatted_results += "\n" + \
                "\n".join(f"- {doc}" for doc in graph_res)
            new_entities = [
                entity for entity in response_data['entities'] if entity not in used_entities]
            used_entities.update(new_entities)
        graph_res = []
        for entity in used_entities:
            graph_res.append(self.graph_manager.graph.nodes[entity])
        formatted_results = self.get_formatted_documents_and_metadatas(
            results, graph_res)
        return formatted_results, 'yes', response_data['entities'], response_data['reasoning']
        return formatted_results

    def search_only_graph(self, query, user_id=None, max_hops=4):
        """
        Search for memories based on a query in the vector store with multi-hopping up to 4 times.
        """
        all_nodes = list(self.graph_manager.graph.nodes)
        graph_res = []
        used_entities = set()
        # Entity extraction prompt
        entity_extraction_prompt = f"""
            Analyze the following text and extract all relevant entities. Each entity should be specific and atomic, such as a Person, Location, Organization, Event, Concept, or Object.
            Allowed entities: {all_nodes}
            Previously used entities: {list(used_entities)}
            Guidelines:
            1. Focus on extracting entities that are directly mentioned or strongly implied in the text.
            2. Avoid overly broad or generic terms.
            3. Include only entities that are likely to be useful for answering the query.
            4. If an entity is not in the allowed list but is crucial to the query, include it anyway.
            5. Prioritize entities that haven't been used before.
            Text: {query}
            Provide your answer in the following JSON format:
            {{
                "entities": ["entity1", "entity2", ...]
            }}
            """

        response = client.chat.completions.create(
            messages=[
                {"role": "user", "content": entity_extraction_prompt}],
            model="gpt-4o-mini",
            response_format={"type": "json_object"}
        )

        ner = json.loads(response.choices[0].message.content)['entities']
        # print('Extracted Entities:', ner)

        new_entities = [
            entity for entity in ner if entity not in used_entities]
        used_entities.update(new_entities)

        for entity in new_entities:
            if entity in self.graph_manager.graph.nodes:
                graph_res.append(
                    entity + ": " + str(self.graph_manager.graph.nodes[entity]))
                graph_res.append(self.graph_manager.get_node_edges(entity))

        formatted_results = "\n\nGraph Results:\n" + \
            "\n".join(f"- {doc}" for doc in graph_res)
        # print(f"AFTER INITIAL QUERY: {formatted_results}")
        analysis_prompt = f"""
            You are an AI assistant tasked with determining if more information is needed to answer a query based on provided information from a Knowledge Graph.
            The primary question is, "Do you have enough information to answer the query based on the provided information?"
            Query: {query}
            Current Information: {formatted_results}
            Allowed Entities: {all_nodes}
            Previously Used Entities: {list(used_entities)}
            Instructions:
            1. Carefully analyze the query and the provided information.
            2. Determine if you have enough information to answer the query or if more hops are needed (and allowed).
            3. If you need additional information and more hops are allowed, specify which entities you need more details about.
            4. Only include entities that are in the Allowed Entities list and not in the Previously Used Entities list.
            5. If the current Information does not have enough information to answer the Query, you should request more information.
            6. Do not use your own knowledge to try and answer the question. Ensure that if you answer 'no', the Current Information is enough to answer the Query. 
            7. You should respond with your reasoning for your answer. This reasoning should be based on why you claim the current information can answer the query. 
            8. You should also create a sub-query, a query which must be answered next, or a simplified version of the current query given the current information.
            Respond in JSON format as follows:
            9. You may not output entities that are previously used. Hence, if you believe that the information may not be present, you may answer 'yes', i.e. if you believe all relevant entities have been accessed. 
            {{
                "response": "yes" or "no",
                "entities": ["entity1", "entity2", ...] (only if more information is needed, otherwise an empty list),
                "reasoning": "Reasoning for your response
                "sub-query": "Sub-query to be answered
            }}
        """
        messages = []
        for hop in range(max_hops):
            # Analysis prompt
            messages.append({"role": "user", "content": analysis_prompt})
            response = client.chat.completions.create(
                messages=messages,
                model="gpt-4o-mini",
                response_format={"type": "json_object"}
            )
            messages.append(
                {"role": "assistant", "content": response.choices[0].message.content})
            # messages.append
            response_data = json.loads(response.choices[0].message.content)
            # print(response_data)
            if response_data['response'] == 'yes' or hop == max_hops - 1:
                # print(f"Multi-hop search completed after {hop + 1} hops.")

                # return formatted_results, 'no', [], response_data['reasoning']
                return formatted_results

            # print(f"HOP {hop + 1} COMPLETED - MORE INFORMATION NEEDED")
            # print("Additional entities requested:", response_data['entities'])
            graph_res = []
            for entity in response_data['entities']:
                try:
                    graph_res.append(self.graph_manager.graph.nodes[entity])
                except Exception as e:
                    pass
                graph_res.append(self.graph_manager.get_node_edges(entity))
            formatted_results += "\n" + \
                "\n".join(f"- {doc}" for doc in graph_res)
            new_entities = [
                entity for entity in response_data['entities'] if entity not in used_entities]
            used_entities.update(new_entities)

        return formatted_results, 'yes', response_data['entities'], response_data['reasoning']
    # def search_only_graph(self, query, user_id=None):
    #     """
    #     Search for memories based on a query in the vector store.
    #     """
    #     all_nodes = list(self.graph_manager.graph.nodes)

    #     # Enhanced first prompt
    #     entity_extraction_prompt = f"""
    #     Analyze the following text and extract all relevant entities. Each entity should be specific and atomic, such as a Person, Location, Organization, Event, Concept, or Object.

    #     Allowed entities: {all_nodes}

    #     Guidelines:
    #     1. Focus on extracting entities that are directly mentioned or strongly implied in the text.
    #     2. Avoid overly broad or generic terms.
    #     3. Include only entities that are likely to be useful for answering the query.
    #     4. If an entity is not in the allowed list but is crucial to the query, include it anyway.

    #     Text: {query}

    #     Provide your answer in the following JSON format:
    #     {{
    #         "entities": ["entity1", "entity2", ...]
    #     }}
    #     """

    #     response = client.chat.completions.create(
    #         messages=[{"role": "user", "content": entity_extraction_prompt}],
    #         model="gpt-4o-mini",
    #         response_format={"type": "json_object"}
    #     )

    #     ner = json.loads(response.choices[0].message.content)['entities']
    #     print('Extracted Entities:', ner)

    #     graph_res = []
    #     for entity in ner:
    #         graph_res.append(self.graph_manager.graph.nodes[entity])
    #         graph_res.append(self.graph_manager.get_node_edges(entity))
    #     formatted_results = "\n\nGraph Results:\n" + \
    #         "\n".join(f"- {doc}" for doc in graph_res)

    #     print("AFTER FIRST QUERY ",formatted_results)
    #     # Enhanced second prompt
    #     analysis_prompt = f"""
    #     You are an AI assistant tasked with determining if more information is needed to answer a query based on provided information from a Knowledge Graph.

    #     Query: {query}
    #     Related Information: {formatted_results}
    #     Allowed Entities: {all_nodes}
    #     Previously Used Entities: {ner}

    #     Instructions:
    #     1. Carefully analyze the query and the provided information.
    #     2. Determine if you have enough information to answer the query.
    #     3. If you need additional information, specify which entities you need more details about.
    #     4. Only include entities that are in the Allowed Entities list and not in the Previously Used Entities list.

    #     Respond in JSON format as follows:
    #     {{
    #         "response": "yes" or "no",
    #         "entities": ["entity1", "entity2", ...] (only if more information is needed, otherwise an empty list)
    #     }}
    #     """

    #     response = client.chat.completions.create(
    #         messages=[{"role": "user", "content": analysis_prompt}],
    #         model="gpt-4o-mini",
    #         response_format={"type": "json_object"}
    #     )

    #     response_data = json.loads(response.choices[0].message.content)

    #     if response_data['response'] == 'yes':
    #         print("MULTIHOP ACTIVATED")
    #         print("Additional entities requested:", response_data['entities'])
    #         for entity in response_data['entities']:
    #             graph_res.append(self.graph_manager.graph.nodes[entity])
    #             graph_res.append(self.graph_manager.get_node_edges(entity))
    #         formatted_results += "\n" + "\n".join(f"- {doc}" for doc in graph_res)

    #     return formatted_results, response_data['response'], response_data['entities']
    #     #
    #     # return formatted_results, response_data.get('answer'), response_data.get('reasoning')

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
