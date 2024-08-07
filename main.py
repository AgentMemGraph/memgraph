from openai import OpenAI
from pprint import pprint
from termcolor import colored, cprint
import sys
from datasets import load_dataset
from memgraph import MemgraphMemory
m = MemgraphMemory()


client = OpenAI()


ds = load_dataset("dgslibisey/MuSiQue")
train_ds = ds["validation"]
for i in range(50, 60):
    m.reset()
    question = train_ds[i]["question"]
    context = train_ds[i]["paragraphs"]

    for paragraph in context:
        doc_id = m.add(paragraph['title'] + "\n" + paragraph['paragraph_text'], user_id="gabe",
                       metadata={}, update=False)
    answer = train_ds[i]["answer"]
    print(f"Question: {question}")
    # print(f"Context: {total_context}")
    # doc_id = m.add(total_context, user_id="gabe", metadata={}, update=False)
    print(f"Stored memory ID: {doc_id}")
    print("All Relationships: \n")
    all_relationships = m.graph_manager.get_all_relationships()
    for relationship in all_relationships:
        print(relationship)
    related_memories1 = m.search_only_graph(query=question, user_id="gabe")
    pprint("Related Memories for Query: \n" + str(related_memories1))
    prompt = (
        f"You are given a query and information that can be used to answer the query. Answer the Query. There are also a set of Entity Triplets given. Use those to make inferences. The answer may not be direct. Do not use your own knowledge, only use answer using the Information provided. If the answer is not presnt in the information provided, output 'Not Enough Information' \n Query: {question} \n Information: {related_memories1} Answer: ")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt},
        ]
    )
    print("====== IMPORTANT ========")
    # print(f"Answer: {answer}")
    cprint(f"Answer: {answer}", "red")
    cprint(response.choices[0].message.content, "red")
    # print(response.choices[0].message.content)
    print("=====================================================")
    print()

# Store a Memory
# Adding all texts to the memory store
# texts = [
#     "John went to the park today and met his old friend, Sarah.",
#     "In the afternoon, John attended a meeting with his boss about the upcoming project.",
#     "John wrote in his diary about feeling anxious before his presentation tomorrow.",
#     "Sarah told John about her plans to travel to Europe next summer.",
#     "John decided to start a new hobby—painting landscapes.",
#     "After the meeting, John had dinner with his colleague, James.",
#     "John's sister called to inform him about the family reunion next month.",
#     "John spent the evening reading a book on mindfulness.",
#     "John helped his neighbor fix a flat tire on their car.",
#     "John received an invitation to his high school reunion.",
#     "John and Sarah went hiking over the weekend.",
#     "John felt relieved after finishing his project ahead of the deadline.",
#     "John's dog, Max, was not feeling well, so he took him to the vet.",
#     "John attended a webinar on digital marketing strategies.",
#     "John ran into an old classmate at the grocery store.",
#     "John spent the weekend at a cabin in the mountains.",
#     "John's boss praised him for his hard work on the recent project.",
#     "John bought a new set of paints to explore his artistic side.",
#     "John and James discussed their favorite books over lunch.",
#     "John spent the afternoon organizing his workspace.",
#     "John visited a local art gallery to find inspiration for his paintings.",
#     "John's parents sent him a care package with his favorite snacks.",
#     "John had a long conversation with Sarah about their future plans.",
#     "John was thrilled to receive a promotion at work.",
#     "John took Max for a walk in the nearby nature reserve.",
#     "John tried a new recipe for dinner and was happy with the results.",
#     "John attended a yoga class to help reduce stress.",
#     "John and Sarah watched a documentary on wildlife conservation.",
#     "John volunteered at a local charity event over the weekend.",
#     "John was invited to speak at a conference about his work.",
#     "John bought tickets to a concert happening next month.",
#     "John's sister visited him for the weekend, and they spent time reminiscing.",
#     "John and his friends planned a weekend getaway to the beach.",
#     "John was nervous about an upcoming presentation at work.",
#     "John took a day off to spend time painting in his studio.",
#     "John attended a networking event and made several new connections.",
#     "John had a productive day and completed all his tasks ahead of time.",
#     "John visited a new coffee shop that opened in his neighborhood.",
#     "John decided to start journaling every night before bed.",
#     "John and Sarah attended a wedding of a mutual friend.",
#     "John spent the afternoon researching new techniques for his paintings.",
#     "John was delighted to receive a surprise gift from Sarah.",
#     "John and James started working on a joint project at work.",
#     "John visited his parents for a family dinner.",
#     "John spent the evening catching up on his favorite TV series.",
#     "John's sister asked him to be the godfather to her newborn son.",
#     "John attended a photography workshop to improve his skills.",
#     "John took a long walk along the beach to clear his mind.",
#     "John and Sarah decided to plan a trip together next year.",
#     "John received positive feedback on his recent work from his team."
# ]

# Adding all texts to memory
# for text in texts:
#     doc_id = m.add(text, user_id="john", metadata={})
# #     print(f"Stored memory ID: {doc_id}")

# # Search for specific queries
# query1 = "What did John Watch and what was it about?"
# query2 = "Where did John spend the weekend?"

# related_memories1 = m.search(query=query1, user_id="john")
# prompt = (f"You are given a query and information that can be used to answer the query. Answer the Query. There are also a set of Entity Triplets given. Use those to make inferences. The answer may not be direct. Query: {query1} \n Information: {related_memories1} Answer: ")

# response = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=[
#         {"role": "system", "content": prompt},
#     ]
# )
# pprint(response.choices[0].message.content)
# pprint(related_memories1)
# # print("Related Memories for Query 1:", related_memories1)

# # related_memories2 = m.search(query=query2, user_id="john")
# pprint(m.graph_manager.get_all_relationships())
# # print("Related Memories for Query 2:", related_memories2)
