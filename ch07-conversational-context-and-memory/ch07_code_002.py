import os
from operator import itemgetter
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)

memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="chat_history"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

runnable = RunnablePassthrough.assign(
    chat_history=(
        RunnableLambda(memory.load_memory_variables)
        | itemgetter("chat_history")
    )
)

chain = runnable | prompt | model

def main():
    print("Type exit or quit to exit the chat")
    try:
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in {"exit", "quit"}:
                print("Exiting......")
                break
            if not user_input:
                continue
            
            response = chain.invoke({"input": user_input})
            
            # Save context to memory
            memory.save_context({"input": user_input}, {"output": response.content})
            
            print("AI:", response.content)
            print("=" * 58)
    except KeyboardInterrupt:
        print("\nExiting......")

if __name__ == '__main__':
    main()
