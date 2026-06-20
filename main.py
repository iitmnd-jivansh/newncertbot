from rag import ask

print("SSC RAG Assistant")
print("Type 'exit' to quit\n")

while True:
    question = input("You: ")

    if question.lower() in ["exit", "quit"]:
        break

    try:
        answer = ask(question)

        print("\nAssistant:")
        print(answer)
        print()

    except Exception as e:
        print("\nERROR:")
        print(e)
        print()
