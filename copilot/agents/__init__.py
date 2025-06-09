import os
from .registry import ModelRegistry, AgentRegistry, DynamicQueryEngineTool, register_agent

OLLAMA_HOST_LLM = os.environ.get('OLLAMA_HOST_LLM', 'localhost')
OLLAMA_PORT_LLM = os.environ.get('OLLAMA_PORT_LLM', 11434)
OLLAMA_HOST_MLLM = os.environ.get('OLLAMA_HOST_MLLM', 'localhost')
OLLAMA_PORT_MLLM = os.environ.get('OLLAMA_PORT_MLLM', 11434)
OLLAMA_SERVER_LLM = f'http://{OLLAMA_HOST_LLM}:{OLLAMA_PORT_LLM}'
OLLAMA_SERVER_MLLM = f'http://{OLLAMA_HOST_MLLM}:{OLLAMA_PORT_MLLM}'

#def llama_index_auto_load_llm_model(model='gpt-3.5-turbo', **kwargs):
#    try:
#        from llama_index.llms.openai import OpenAI
#        llm = OpenAI(model=model, **kwargs)
#    except:
#        from llama_index.llms.ollama import Ollama
#        llm = Ollama(
#            model=model, 
#            base_url=OLLAMA_SERVER_LLM, 
#            **kwargs,
#        )

#    return llm

def llama_index_auto_load_llm_model(model='gpt-3.5-turbo', **kwargs):
    ollama_model_map = {
        'llava-13b': 'llava:13b-v1.6-vicuna-q8_0',  # adjust based on your Ollama server config
        'llama3': 'llama3.1:latest',
    }
    if model in ollama_model_map:
        model = ollama_model_map[model]
        print("mapping {model}")
    try:
        from llama_index.llms.openai import OpenAI
        if "gpt" in model:
            print("llm load openai model")
            return OpenAI(model=model, **kwargs)
        else:
            print(f"Model '{model}' not recognized for OpenAI. Falling back to Ollama.")
            raise ImportError("Not an OpenAI model")
    except ImportError as e:
        from llama_index.llms.ollama import Ollama
        print("llm load ollama model")
        try:
            llm = Ollama(
                model=model, 
                base_url=OLLAMA_SERVER_LLM, 
                **kwargs,
            )
            print(f"Successfully loaded Ollama model: {model}")
            return llm
        except Exception as ex:
            print(f"Failed to load Ollama model '{model}': {ex}")
            raise ex
from .rag import RAGRouter
from .mllm_clients import OllamaClient, GPTClient, resize_pil
from .llm_retriver import build_vector_store_index, get_query_engine
from .tme_features import *

# Export REGISTRY and register_agent so they can be imported in other modules
__all__ = [
    'AgentRegistry', 
    'ModelRegistry',
    'DynamicQueryEngineTool', 
    'OllamaClient',
    'GPTClient',
    'RAGRouter', 
    'register_agent', 
    'llama_index_auto_load_llm_model',
    'build_vector_store_index',
    'get_query_engine',
    'resize_pil',
]
