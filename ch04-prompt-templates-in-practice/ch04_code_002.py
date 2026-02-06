
from langchain_classic import hub

# Get the latest version of a prompt
# prompt = hub.pull("rlm/rag-prompt")
prompt = hub.pull("rlm/rag-prompt:50442af1")

print(prompt)