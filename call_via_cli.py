import argparse
from openai import OpenAI
from memgraph import MemgraphMemory
import json
from pprint import pprint
client = OpenAI()
m = MemgraphMemory()
# m.reset()

def chat_with_memory(query, chat_history):
    # Search memory
    related_memories = m.search(query=query, user_id="user")
    pprint(related_memories)
    # Prepare the prompt with memory context
    prompt = f"""You are an AI assistant with access to previous conversations and related information. 
    Use the following information to inform your response, but don't explicitly mention it unless relevant:

    Related Information: {related_memories}

    Chat History:
    {chat_history}

    Human: {query}
    AI Assistant:"""

    # Query ChatGPT
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()


def main():
    parser = argparse.ArgumentParser(
        description="Chat with GPT-4 using memory.")
    parser.add_argument("--clear", action="store_true",
                        help="Clear the memory before starting")
    args = parser.parse_args()

    if args.clear:
        m.graph_manager.clear_graph()
        print("Memory cleared.")

    chat_history = []

    print("Welcome to the memory-enhanced chat! Type 'exit' to quit.")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == 'exit':
            break

        # Get response from ChatGPT with memory context
        ai_response = chat_with_memory(user_input, "\n".join(chat_history))

        print(f"AI: {ai_response}")

        # Update chat history
        chat_history.append(f"Human: {user_input}")
        chat_history.append(f"AI: {ai_response}")

        # Add the exchange to memory
        memory_entry = f"Human: {user_input}\nAI: {ai_response}"
        doc_id = m.add(memory_entry, user_id="user", metadata={})
        # print(f"Memory stored with ID: {doc_id}")


if __name__ == "__main__":
    main()
