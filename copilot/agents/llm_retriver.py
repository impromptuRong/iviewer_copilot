from llama_index.core import Document, VectorStoreIndex
from . import register_agent


description = f"""\
    The agent build vector store index for description. 
"""
@register_agent(
    name='description_vector_index',
    type=None,  # we treat the function as hidden
    input_mapping={
        'text': 'description',
    },
    output_mapping='description_vector_store_index',
    description=description,
)
def build_vector_store_index(text):
    documents = [Document(text=t) for t in text]
    # splitter = SentenceSplitter(chunk_size=1024)
    # nodes = splitter.get_nodes_from_documents(documents)
    # index = VectorStoreIndex(nodes)
    index = VectorStoreIndex.from_documents(documents)

    return index


description = f"""\
    You are tasked with retrieving accurate and relevant information from existing description based on the user's prompt. 
    The description may cover additional information about secondary structure, architecture, growth pattern. 
    Provide information that cannot be discovered from other functional agent. 
"""
@register_agent(
    name='description_query_engine',
    type='DynamicQueryEngineTool',
    input_mapping={
        'vector_store_index': 'description_vector_store_index',
    },
    output_mapping='description_query_engine',
    description=description,
)
def get_query_engine(vector_store_index):
    return vector_store_index.as_query_engine()
