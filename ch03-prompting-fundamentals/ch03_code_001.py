
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)

examples = [
    {
        "question": "Who lived longer, Steve Jobs or Albert Einstein?",
        "answer": """Do we need additional questions? Yes.
Steve Jobs died at age 56.
Albert Einstein died at age 76.
Final answer: Albert Einstein."""
    },
    {
        "question": "When was Google's founder born?",
        "answer": """Do we need additional questions? Yes.
Google was founded by Larry Page.
Larry Page was born on March 26, 1973.
Final answer: March 26, 1973."""
    },
]

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples=examples,
    embeddings=embeddings,
    vectorstore_cls=Chroma,
    k=1,
)

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{instruction}\n{input}"),
        ("ai", "{answer}"),
    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        few_shot_prompt,
        ("human", "{instruction}\n{input}"),
    ]
)

chain = final_prompt | llm

response = chain.invoke({
    "instruction": "Please write meeting minutes",
    "input": "On December 26, the product team reviewed project progress..."
})

print(response.content)

