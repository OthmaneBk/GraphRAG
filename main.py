import os
import time
from langchain_community.graphs import MemgraphGraph
from langchain_experimental.graph_transformers.llm import LLMGraphTransformer
from langchain_core.documents import Document
from langchain_community.chat_models import ChatOllama

from langchain_text_splitters import CharacterTextSplitter
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain

url = os.environ.get("MEMGRAPH_URI", "bolt://localhost:7687")
username = os.environ.get("MEMGRAPH_USERNAME", "")
password = os.environ.get("MEMGRAPH_PASSWORD", "")



#initialize memgraph connection
graph = MemgraphGraph(
    url=url, username=username, password=password, refresh_schema=True
)


graph_text = """

John's title is Director of the Digital Marketing Group.

John works with Jane whose title is Chief Marketing Officer.

Jane works in the Executive Group.

Jane works with Sharon whose title is the Director of Client Outreach.

Sharon works in the Sales Group.
"""



llm = ChatOllama(
    model="llama3.1:latest",
    temperature=0,
    num_predict=1000,  # Limite maximale de tokens à générer
    top_k=10,
    top_p=0.8
)

llm_transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=["Person", "Title", "Group"],
    allowed_relationships=["TITLE", "COLLABORATES", "GROUP"]
)

text_splitter = CharacterTextSplitter(
    separator=".",
    chunk_size=1,        # Taille minimale pour forcer la séparation à chaque point
    chunk_overlap=0,     # Pas de chevauchement nécessaire ici
    is_separator_regex=False
)


documents = text_splitter.create_documents([graph_text])

graph_documents = llm_transformer.convert_to_graph_documents(documents)


#print("graph_documents:", graph_documents)



graph.query("STORAGE MODE IN_MEMORY_ANALYTICAL")
graph.query("DROP GRAPH")
graph.query("STORAGE MODE IN_MEMORY_TRANSACTIONAL")


#graph.add_graph_documents(graph_documents)
graph.refresh_schema()

#print(graph.get_schema)


# 2. Exemples simples (sans tokens spéciaux manuels ni doubles accolades)
examples = [
    {
        "question": "What group is Charles in?",
        "query": "MATCH (p:Person {{id: 'Charles'}})-[:GROUP]->(g) RETURN g.id",
        "response": "Sales Department",
    },  
    {
        "question": "Who does Paul work with?",
        "query": "MATCH (a:Person {{id: 'Paul'}})-[:COLLABORATES]->(p) RETURN p.id",
        "response": "Jane",
    },
    {
        "question": "What title does Rico have?",
        "query": "MATCH (p:Person {{id: 'Rico'}})-[:TITLE]->(t) RETURN t.id",
        "response": "Director of Client Outreach",
    },
]

# 3. Prompt d'exemple épuré
example_prompt = PromptTemplate(
    input_variables=["question", "query", "response"],
    template="Question: {question}\nCypher: {query}\nResult: {response}",
)

# 4. Instructions
prefix = """You are a Cypher expert for Memgraph.
Given an input question and schema, write a Cypher query.
Output ONLY the Cypher query and nothing else.

Database Schema:
{schema}"""

cypher_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix="Question: {question}\nCypher: ",
    input_variables=["question", "schema"],
)

# 5. Exécution automatique via la chaîne (Recommandé)
chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    cypher_prompt=cypher_prompt,
    verbose=True,
    allow_dangerous_requests=True,
)

response = chain.invoke({"query": "What title does John have?"})
print(response["result"])