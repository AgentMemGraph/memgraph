import chromadb


class ChromaDBManager:
    def __init__(self, collection_name):
        # Initialize ChromaDB client
        self.client = chromadb.Client()
        self.collection_name = collection_name
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        try:
            # Attempt to get the collection, create it if it does not exist
            collection = self.client.get_or_create_collection(
                self.collection_name)
        except Exception as e:
            raise RuntimeError(f"Error accessing or creating collection: {e}")
        return collection

    def add_documents(self, documents, metadatas=None, ids=None):
        try:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print("Documents added successfully.")
        except Exception as e:
            raise RuntimeError(f"Error adding documents: {e}")

    def update_document(self, doc_id, document, metadata=None):
        try:
            # Update document - Delete old and add new
            self.delete_document(doc_id)
            self.add_documents([document], [metadata], [doc_id])
            print(f"Document with ID '{doc_id}' updated successfully.")
        except Exception as e:
            raise RuntimeError(f"Error updating document: {e}")

    def delete_document(self, doc_id):
        try:
            self.collection.delete(ids=[doc_id])
            print(f"Document with ID '{doc_id}' deleted successfully.")
        except Exception as e:
            raise RuntimeError(f"Error deleting document: {e}")

    def query(self, query_texts, n_results=5, where=None, where_document=None):
        try:
            results = self.collection.query(
                query_texts=query_texts,
                n_results=n_results,
                where=where,
                where_document=where_document
            )
            return results
        except Exception as e:
            raise RuntimeError(f"Error querying documents: {e}")

    def list_documents(self):
        try:
            # This is a placeholder function; ChromaDB does not have a direct method to list all documents
            raise NotImplementedError(
                "Listing documents is not directly supported.")
        except Exception as e:
            raise RuntimeError(f"Error listing documents: {e}")

    def get_document_by_id(self, doc_id):
        try:
            # Query by ID
            results = self.collection.get(ids = [doc_id])
            return results
        except Exception as e:
            raise RuntimeError(f"Error retrieving document by ID: {e}")

    def get_all(self):
        try:
            # Get all document IDs
            documents = self.collection.get()
            return documents
        except Exception as e:
            raise RuntimeError(f"Error retrieving all documents: {e}")
        
    def delete_all(self):
        try:
            # Delete all documents
            self.client.delete_collection(self.collection_name)
            self.collection = self._get_or_create_collection()
        except Exception as e:
            raise RuntimeError(f"Error deleting all documents: {e}")
        
# Example usage:
if __name__ == "__main__":
    manager = ChromaDBManager("all-my-documents")

    # Add documents
    manager.add_documents(
        documents=["This is document1", "This is document2"],
        metadatas=[{"source": "notion"}, {"source": "google-docs"}],
        ids=["doc1", "doc2"]
    )

    # Query documents
    results = manager.query(
        query_texts=["This is a query document"], n_results=2)
    print(results)

    # Update document
    manager.update_document("doc1", "Updated content for document1", {
                            "source": "notion-updated"})

    # Delete document
    manager.delete_document("doc2")
