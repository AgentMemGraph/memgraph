from memgraph import MemgraphMemory
m = MemgraphMemory()

# Store a Memory
doc_id = m.add("Likes to play cricket on weekends",
               user_id="alice", metadata={"category": "hobbies"})
print(f"Stored memory ID: {doc_id}")

# Retrieve Memories
all_memories = m.get_all()
print("All Memories:", all_memories)

# Retrieve a single memory by ID
specific_memory = m.get(doc_id)
print("Specific Memory:", specific_memory)

# Search Memories
related_memories = m.search(
    query="What are Alice's hobbies?", user_id="alice")
print("Related Memories:", related_memories)

# Update a Memory
updated_id = m.update(
    memory_id=doc_id, data="Likes to play tennis on weekends")
print(f"Updated memory ID: {updated_id}")

# Memory History (if implemented)
# history = m.history(memory_id=doc_id)
# print("Memory History:", history)

# Delete Memory
m.delete(memory_id=doc_id)
print(f"Deleted memory ID: {doc_id}")

# Delete all memories for a user
m.delete_all(user_id="alice")
print("Deleted all memories for user 'alice'")

# Reset all memories
m.reset()
print("All memories have been reset.")
