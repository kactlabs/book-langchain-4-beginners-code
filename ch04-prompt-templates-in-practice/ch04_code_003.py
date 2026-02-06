
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic import hub

prompt = ChatPromptTemplate.from_template(
    "Summarize the following text. "
    "Write the summary.\n\n"
    "CONTEXT:\n{context}\n\nSUMMARY:"
)

# from langchain import hub

hub.push("example/simple-summary", prompt)