from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex
from llama_index.core.tools import QueryEngineTool, FunctionTool
from llama_index.core.agent import ReActAgent
# from llama_index.agent.openai import OpenAIAgent
# from llama_index.core.storage.chat_store import SimpleChatStore
# from llama_index.core.tools.utils import create_schema_from_function

from . import AgentRegistry, DynamicQueryEngineTool, llama_index_auto_load_llm_model
from typing import Any, Dict, Optional


# default model "gpt-4o-mini"
class RAGRouter:
    def __init__(
        self, 
        agents: Dict[str, Any], 
        llm: str='gpt-3.5-turbo', 
        similarity_top_k: int=3, 
        llm_cfgs: Optional[Dict]=None
    ) -> None:
        self.tools = {}
        for name, cfg in agents.items():
            self.register_agent(name, **cfg)

        llm_model = llama_index_auto_load_llm_model(model=llm, **(llm_cfgs or {}))
        self.init_rag_engine(llm=llm_model, similarity_top_k=similarity_top_k, )

    @classmethod
    def from_agent_registry(
        cls, 
        agent_registry: AgentRegistry, 
        llm: str='gpt-3.5-turbo', 
        similarity_top_k: int=3, 
        llm_cfgs: Optional[Dict]=None
    ) -> "RAGRouter":
        agents = {}
        for name, cfg in agent_registry.registry.items():
            agent_type = cfg.get('type', None)
            if agent_type == 'DynamicQueryEngineTool':
                entrypoint = agent_registry.get_query_engine(name)
            else:
                entrypoint = agent_registry.get_function(name)
            if agent_type:
                agents[name] = {
                    'type': agent_type,
                    'entrypoint': entrypoint,
                    'description': cfg.get('description', f"This is a agent for {name}"),
                    'return_direct': cfg.get('return_direct', False),
                }

        return cls(agents, llm=llm, similarity_top_k=similarity_top_k, llm_cfgs=llm_cfgs)

    def register_agent(self, name: str, type: str, entrypoint: Any, description: str='', return_direct: bool=False, **kwargs) -> None:
        assert name not in self.tools, f"Agent:{name} already exists in RAGRouter. "
        if not description:
            print(f"Agent:{name} has empty description. ")

        # fn_schema = create_schema_from_function(agent_name, entrypoint, additional_fields=None)
        # We create a placeholder function to hack the parameters. 
        if type == 'FunctionTool':
            tool = FunctionTool.from_defaults(
                name=name, fn=entrypoint,
                description=description,
                return_direct=return_direct,
            )
        elif type == 'QueryEngineTool':
            tool = QueryEngineTool.from_defaults(
                name=name, query_engine=entrypoint,
                description=description,
                return_direct=return_direct,
            )
        elif type == 'DynamicQueryEngineTool':
            tool = DynamicQueryEngineTool.from_defaults(
                name=name, query_engine=entrypoint,
                description=description,
                return_direct=return_direct,
            )
        else:
            raise ValueError(f"Got agent type: {type} (must be one of ['FunctionTool', 'QueryEngineTool', 'DynamicQueryEngineTool']). ")

        self.tools[name] = tool

    def init_rag_engine(self, llm=None, similarity_top_k: Optional[int]=None) -> None:
        self.similarity_top_k = similarity_top_k or self.similarity_top_k
        self.llm = llm or self.llm
        print(f"Initial RAG engine with llm: {self.llm}")

        self.obj_index = ObjectIndex.from_objects(
            self.tools.values(),
            index_cls=VectorStoreIndex,
        )
        self.tool_retriever = self.obj_index.as_retriever(similarity_top_k=self.similarity_top_k)
        # self.chat_store = SimpleChatStore()
        self.agent = ReActAgent.from_tools(  # OpenAIAgent
            tool_retriever=self.tool_retriever, 
            llm=self.llm, 
            # memory=self.chat_store,
            verbose=True
        )
