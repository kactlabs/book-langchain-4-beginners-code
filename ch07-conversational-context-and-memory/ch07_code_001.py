
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# model = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     temperature=0.2
# )

# response = model.invoke(
#     "Explain artificial intelligence in one sentence."
# )

# print(response.content)

from langchain_community.chat_message_histories import SQLChatMessageHistory

chat_message_history = SQLChatMessageHistory(
    session_id="sql_history",
    connection="sqlite:///sqlite.db"
)

# Add a user message
chat_message_history.add_user_message(
    "Hello! Nice to meet you. My name is Teddy."
)

# Add an AI message
chat_message_history.add_ai_message(
    "Hi Teddy, nice to meet you as well!"
)

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

# from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

chain = prompt | ChatGoogleGenerativeAI(model="gemini-2.5-flash") | StrOutputParser()

from langchain_community.chat_message_histories import SQLChatMessageHistory

def get_chat_history(user_id, conversation_id):
    return SQLChatMessageHistory(
        table_name=user_id,
        session_id=conversation_id,
        connection="sqlite:///sqlite.db",
    )


from langchain_core.runnables.utils import ConfigurableFieldSpec

config_fields = [
    ConfigurableFieldSpec(
        id="user_id",
        annotation=str,
        name="User ID",
        description="Unique identifier for a user.",
        is_shared=True,
    ),
    ConfigurableFieldSpec(
        id="conversation_id",
        annotation=str,
        name="Conversation ID",
        description="Unique identifier for a conversation.",
        is_shared=True,
    ),
]

from langchain_core.runnables.history import RunnableWithMessageHistory

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_chat_history,
    input_messages_key="question",
    history_messages_key="chat_history",
    history_factory_config=config_fields,
)

config = {
    "configurable": {
        "user_id": "user1",
        "conversation_id": "conversation1"
    }
}

chain_with_history.invoke(
    {"question": "Hello, my name is Teddy."},
    config
)

config = {
    "configurable": {
        "user_id": "user1",
        "conversation_id": "conversation2"
    }
}

result = chain_with_history.invoke(
    {"question": "What is my name?"},
    config
)

print(result)

