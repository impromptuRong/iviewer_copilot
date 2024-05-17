import os
import config

from mllm_clients import OllamaClient, GPTClient

class ModelRegistry:
    __entry__ = ['caption', 'chatbot']
    def __init__(self):
        self._registry = {k: {} for k in self.__entry__}
    
    def register(self, name, entry, client):
        assert entry in self.__entry__
        self._registry[entry][name] = client

    def get_caption_model(self, name):
        return self._registry['caption'].get(name)
    
    def get_chatbot_model(self, name):
        return self._registry['chatbot'].get(name)
    
    def info(self):
        return self._registry


MODEL_REGISTRY = ModelRegistry()

ollama_host = os.environ.get('OLLAMA_HOST', 'localhost')
ollama_port_caption = os.environ.get('OLLAMA_PORT_CAPTION', 11434)
ollama_port_chatbot = os.environ.get('OLLAMA_PORT_CHATBOT', 11435)

## LLaVA + LLaMA3
llava_caption_client = OllamaClient(
    config.llava, host=f'http://{ollama_host}:{ollama_port_caption}',
)

llama3_chatbot_client = OllamaClient(
    config.llama3, host=f'http://{ollama_host}:{ollama_port_chatbot}',
)

MODEL_REGISTRY.register("llava", "caption", llava_caption_client)
MODEL_REGISTRY.register("llava", "chatbot", llama3_chatbot_client)


## GPT4v + GPT3.5
gpt4v_caption_client = GPTClient(config.gpt4v)
gpt35_chatbot_client = GPTClient(config.gpt35_chat)

MODEL_REGISTRY.register("gpt4v", "caption", gpt4v_caption_client)
MODEL_REGISTRY.register("gpt4v", "chatbot", gpt35_chatbot_client)


## GPT4o
gpt4o_caption_client = GPTClient(config.gpt4o)
gpt4o_chatbot_client = GPTClient(config.gpt4o_chat)

MODEL_REGISTRY.register("gpt4o", "caption", gpt4o_caption_client)
MODEL_REGISTRY.register("gpt4o", "chatbot", gpt4o_chatbot_client)
