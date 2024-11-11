import inspect
from collections import deque

from pydantic import BaseModel, Field, create_model, field_validator
from typing import Any, Callable, Dict, List, Literal, Union, Iterable, Optional

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata, ToolOutput


class ModelRegistry:
    __entry__ = ['caption', 'chatbot', 'rag']
    def __init__(self):
        self._registry = {k: {} for k in self.__entry__}

    def register(self, entry, name, client):
        assert entry in self.__entry__
        self._registry[entry][name] = client

    def get_caption_model(self, name):
        return self._registry['caption'].get(name)

    def get_chatbot_model(self, name):
        return self._registry['chatbot'].get(name)

    def get_rag_model(self, name):
        return self._registry['rag'].get(name)

    def info(self):
        return self._registry


DEFAULT_NAME = "query_engine_tool"
DEFAULT_DESCRIPTION = """Useful for running a natural language query
against a knowledge base and get back a natural language response.
"""
class DynamicQueryEngineTool(QueryEngineTool):
    """Dynamic Query engine tool.

    A tool making use of a query engine.

    Args:
        query_engine (function): A function that return a query engine (BaseQueryEngine).
        metadata (ToolMetadata): The associated metadata of the query engine.
    """

    def __init__(
        self,
        query_engine: Callable[..., Any],
        metadata: ToolMetadata,
        resolve_input_errors: bool = True,
    ) -> None:
        self._query_engine = query_engine
        self._metadata = metadata
        self._resolve_input_errors = resolve_input_errors

    @classmethod
    def from_defaults(
        cls,
        query_engine: Callable[..., Any],
        name: Optional[str] = None,
        description: Optional[str] = None,
        return_direct: bool = False,
        resolve_input_errors: bool = True,
    ) -> "DynamicQueryEngineTool":
        name = name or DEFAULT_NAME
        description = description or DEFAULT_DESCRIPTION

        metadata = ToolMetadata(
            name=name, description=description, return_direct=return_direct
        )
        return cls(
            query_engine=query_engine,
            metadata=metadata,
            resolve_input_errors=resolve_input_errors,
        )

    def query_engine(self) -> BaseQueryEngine:
        return self._query_engine()

    def call(self, *args: Any, **kwargs: Any) -> ToolOutput:
        query_str = self._get_query_str(*args, **kwargs)
        response = self.query_engine().query(query_str)
        return ToolOutput(
            content=str(response),
            tool_name=self.metadata.name,
            raw_input={"input": query_str},
            raw_output=response,
        )

    async def acall(self, *args: Any, **kwargs: Any) -> ToolOutput:
        query_str = self._get_query_str(*args, **kwargs)
        response = await self.query_engine().aquery(query_str)
        return ToolOutput(
            content=str(response),
            tool_name=self.metadata.name,
            raw_input={"input": query_str},
            raw_output=response,
        )

    def as_langchain_tool(self):
        from llama_index.core.langchain_helpers.agents.tools import (
            IndexToolConfig,
            LlamaIndexTool,
        )

        tool_config = IndexToolConfig(
            query_engine=self.query_engine(),
            name=self.metadata.name,
            description=self.metadata.description,
        )
        return LlamaIndexTool.from_tool_config(tool_config=tool_config)


class AgentModel(BaseModel):
    fn: Callable[..., Any] = Field(..., description="A callable function with any number of arguments and any return type.")
    name: str = Field(..., description="The name of the function as a string.")
    type: Literal["FunctionTool", "QueryEngineTool"] = Field(..., type=str, description="The type of the agent, must be 'FunctionTool' or 'QueryEngineTool'")
    input_mapping: Dict[Union[int, str], str] = Field(..., description="A dictionary mapping inputs, where keys can be integers or strings, and values are strings.")
    output_mapping: Union[str, List[str], Dict[str, str]] = Field(..., description="The output mapping, which can be a string, a list of strings, or a dictionary of string keys and values.")
    description: str = Field(..., description="A description of the function.")

    class Config:
        arbitrary_types_allowed = True

    @field_validator('fn')
    def validate_function(cls, v):
        if not callable(v):
            raise ValueError('function must be callable')
        return v


def _register_function(registry: Dict[str, Callable[..., Any]], 
                       registered_nodes: Dict[str, Dict], 
                       fn: Callable[..., Any], 
                       fn_name: str, 
                       type: str, 
                       input_mapping: Dict[int | str, str], 
                       output_mapping: str | List[str] | Dict[str, str],
                       description: str,
                       return_direct: bool,
                       ):
        if fn_name in registry:
            raise ValueError(f"{fn_name} is already registered as: {registry[fn_name]}. ")
        # Check whether input_mapping is valid
        for var_name_or_position, node_name in input_mapping.items():
            if node_name not in registered_nodes:
                registered_nodes[node_name] = {'as_fn_input': [], 'as_fn_output': None}
            registered_nodes[node_name]['as_fn_input'].append((fn_name, var_name_or_position))

        # Check whether output_mapping is valid
        if isinstance(output_mapping, str):
            output_mapping = {None: output_mapping}
        elif isinstance(output_mapping, list):
            output_mapping = {idx: node_name for idx, node_name in enumerate(output_mapping) if node_name}
        elif isinstance(output_mapping, dict):
            output_mapping = {k: node_name for k, node_name in output_mapping.items() if node_name}

        for var_name_or_position, node_name in output_mapping.items():
            if node_name not in registered_nodes:
                registered_nodes[node_name] = {'as_fn_input': [], 'as_fn_output': None}
            if registered_nodes[node_name]['as_fn_output'] is not None:
                raise ValueError(f"{node_name} is registered as output by function {registered_nodes[node_name]['as_fn_output']}, change another name. ")
            registered_nodes[node_name]['as_fn_output'] = (fn_name, var_name_or_position)

        # Infer input signature from the function
        sig = inspect.signature(fn)

        input_fields = {}
        for param in sig.parameters.values():
            param_name = param.name
            param_type = param.annotation if param.annotation != inspect.Parameter.empty else Any
            default_value = param.default if param.default != inspect.Parameter.empty else ...
            input_fields[param_name] = (param_type, default_value)
        InputModel = create_model(f"{fn_name}InputModel", **input_fields)

        output_fields = {node_name: (Any, ...) for node_name in output_mapping.values()}
        OutputModel = create_model(f"{fn_name}OutputModel", **output_fields)

        registry[fn_name] = {
            'function': fn,
            'type': type,
            'input_mapping': input_mapping,
            'output_mapping': output_mapping,
            'input_model': InputModel,
            'output_model': OutputModel,
            'signature': sig,
            'description': description,
            'return_direct': return_direct,
        }


class AgentRegistry:
    """ Agent Registry and Dynamic Function Loader
    # Example usage:
    def awesome_add(add_x: int, add_y: int) -> float:
        return add_x + add_y

    def awesome_mul(mul_x: int, mul_z: int = 5) -> float:
        return mul_x * mul_z

    def awesome_div(num1: int, num2: int):
        return num1 / num2, num2 / num1

    registry = AgentRegistry()
    registry.register_function(awesome_add, 'awesome_addition', {}, 'add_result')
    registry.register_function(awesome_mul, 'awesome_multiply', {'mul_x': 'add_result'}, 'mul_result')
    registry.register_function(awesome_div, 'awesome_division', {'num1': 'mul_result', 'num2': 'add_result'}, ['div_forward', 'div_backward'])
    print(registry.registered_nodes)

    # Get functions
    func_add_wrapper = registry.get_function('awesome_addition')
    func_mul_wrapper = registry.get_function('awesome_multiply')
    func_div_wrapper = registry.get_function('awesome_division')

    # Call functions
    print(func_div_wrapper(add_x=5, add_y=1).model_dump())
    print(registry.cache)  # {'add_result': 6, 'mul_result': 30, 'div_forward': 5.0, 'div_backward': 0.2}

    registry.update_cache('add_result', 100)
    print(func_div_wrapper(add_x=5, add_y=1, mul_z=2).model_dump())
    print(registry.cache)  # {'add_result': 100, 'mul_result': 200, 'div_forward': 2.0, 'div_backward': 0.5}
    """
    _registry = {}
    _registered_nodes = {}

    def __init__(self, init_nodes: Optional[Iterable] = None):
        self.registry = self._registry.copy()
        self.registered_nodes = self._registered_nodes.copy()
        self.cache = {}
        if init_nodes:
            for node_name in init_nodes:
                if node_name not in self.registered_nodes:
                    self.registered_nodes[node_name] = {'as_fn_input': [], 'as_fn_output': None}

    @classmethod
    def register_initial_nodes(cls, init_nodes: Optional[Iterable] = None):
        if init_nodes:
            for node_name in init_nodes:
                if node_name not in cls._registered_nodes:
                    cls._registered_nodes[node_name] = {'as_fn_input': [], 'as_fn_output': None}

    @classmethod
    def register_cls_function(cls, fn: Callable[..., Any], name: str, type: str, 
                              input_mapping: Dict[int | str, str], 
                              output_mapping: str | List[str] | Dict[str, str],
                              description: str,
                              return_direct: bool,
                              ):
        _register_function(cls._registry, cls._registered_nodes, fn,
                           fn_name=name, type=type, input_mapping=input_mapping, 
                           output_mapping=output_mapping, description=description,
                           return_direct=return_direct,
                          )

    def register_function(self, fn: Callable[..., Any], name: str, type: str, 
                          input_mapping: Dict[int | str, str], 
                          output_mapping: str | List[str] | Dict[str, str],
                          description: str,
                          return_direct: bool,
                          ):
        _register_function(self.registry, self.registered_nodes, fn, 
                           fn_name=name, type=type, input_mapping=input_mapping, 
                           output_mapping=output_mapping, description=description,
                           return_direct=return_direct,
                          )

    def get_function(self, name: str) -> Callable[..., Any]:
        """
        Return a call function wrapper(**kwargs)
            First check if kwargs contain values with node_name and if we need to update the cache
            for k, v in kwargs.items():
                # this means node_name cannot be calculated and must be provided
                if k in self.registered_nodes and self.registered_nodes[k]['as_fn_output'] is None:
                    self.cache[k] = v
                else: donot update it

            Then find all fn signatures: x, y, var1, var2, kwargs1
            we want to find correct value for each variable: values may have several different scenario
            1. if input_mapping clearly saying x should coming from a node: we use self.cache[node]
                a) if self.cache[node] exist, fn(x=self.cache[node]), regardless whether x is in kwargs
                b) if self.cache[node] doesnot exist, find function
                    if function is not None: then call it.
                    if function is None: that's something need user input, use update_field() #Not implemented yet
            2. if input_mapping didn't say x should coming from a node:
                a) if x in kwargs: we use kwargs[x] because we don't know how to map it to cache.
                b) if x not in kwargs: we try to find default value
                c) if even no default value: that's something need user input, use update_field() #Not implemented yet
        """
        fn_name = name
        if fn_name not in self.registry:
            raise ValueError(f"Function {fn_name} is not registered.")

        fn_info = self.registry[fn_name]
        fn = fn_info['function']
        input_mapping = fn_info['input_mapping']
        output_mapping = fn_info['output_mapping']
        sig = fn_info['signature']
        InputModel = fn_info['input_model']
        OutputModel = fn_info['output_model']

        def wrapper(**kwargs):
            # if all outputs are already in cache, directly return result
            # print(f"-------- Function called: {fn_name}")
            missing_nodes = [node_name for node_name in output_mapping.values() if node_name not in self.cache]
            if not missing_nodes:
                output = {node_name: self.cache[node_name] for node_name in output_mapping.values()}
                print(f"Directly fetch result for {output_mapping.values()} from cache. Skip rerun agent: {fn_name}")
                return output  # OutputModel(**output)
            else:
                print(f"Cannot directly fetch result for {missing_nodes} from cache. Recalculate results for agent: {fn_name}")

            ## LLM tend to auto parse useless information. We explicitly update nodes through update_cache
            for k, v in kwargs.items():
                if k in self.registered_nodes and k not in self.cache and self.registered_nodes[k]['as_fn_output'] is None:
                    # print(f"-------- Adding additional parameters => {k}: {v}")
                    self.cache[k] = v

            # Resolve missing parameters
            fn_kwargs = {}
            for idx, param in enumerate(sig.parameters.values()):
                param_name = param.name
                # print(f"-------- input_mapping: {input_mapping}, {idx}, {param}")
                if idx in input_mapping:
                    node_name = input_mapping[idx]
                    if node_name in self.cache:
                        fn_kwargs[param_name] = self.cache[node_name]
                    else:
                        fn_spec = self.registered_nodes[node_name]['as_fn_output']
                        if fn_spec is not None:
                            dependency_fn_name, _ = fn_spec
                            dependency_fn = self.get_function(dependency_fn_name)
                            _ = dependency_fn(**kwargs)
                            fn_kwargs[param_name] = self.cache[node_name]
                        else:
                            # TODO: return message to chatbox and let user input then update.
                            raise ValueError(f"I cannot get information without the information for variable: {node_name}. ")
                elif param_name in input_mapping:
                    node_name = input_mapping[param_name]
                    if node_name in self.cache:
                        fn_kwargs[param_name] = self.cache[node_name]
                    else:
                        fn_spec = self.registered_nodes[node_name]['as_fn_output']
                        if fn_spec is not None:
                            dependency_fn_name, _ = fn_spec
                            dependency_fn = self.get_function(dependency_fn_name)
                            _ = dependency_fn(**kwargs)
                            fn_kwargs[param_name] = self.cache[node_name]
                        else:
                            # TODO: return message to chatbox and let user input then update.
                            raise ValueError(f"I cannot get answer without additional information for: {node_name}. ")
                else:
                    if param_name in kwargs:
                        fn_kwargs[param_name] = kwargs[param_name]
                    elif param_name:
                        if param.default != inspect.Parameter.empty:
                            fn_kwargs[param_name] = param.default
                        else:
                            # TODO: return message to chatbox and let user input it.
                            raise ValueError(f"I cannot get answer without additional information for: {param_name}. ")
            # print(f"-------- Function {fn_name}, parameters: {fn_kwargs.keys()}")
            # Validate input using Pydantic model and execute the function
            inputs = InputModel(**fn_kwargs)
            result = fn(**inputs.model_dump())

            # Store the result in cache and validate with OutputModel
            for position_or_name, node_name in output_mapping.items():
                if position_or_name is None:
                    self.cache[node_name] = result
                else:
                    self.cache[node_name] = result[position_or_name]

            output = {node_name: self.cache[node_name] for node_name in output_mapping.values()}

            return output  # OutputModel(**output)

        wrapper.__signature__ = fn_info['signature']

        return wrapper

    def get_query_engine(self, name: str) -> BaseQueryEngine:
        fn = self.get_function(name)
        node_name = self.registry[name]['output_mapping'][None]
        def _wrapper(**kwargs):
            # print(f"inner function", name, fn, node_name, r)
            return fn(**kwargs)[node_name]

        return _wrapper

    def update_cache(self, node_name: str, value: Any):
        # all results that rely on modified cache need to be cleaned
        if node_name not in self.registered_nodes:
            return
        self.delete_cache(node_name)
        self.cache[node_name] = value

    def delete_cache(self, node_name: str):
        if node_name not in self.registered_nodes:
            return
        # all results that rely on modified cache need to be cleaned
        pop_nodes = deque([node_name])
        while pop_nodes:
            node_name = pop_nodes.popleft()
            print("poped", node_name)
            self.cache.pop(node_name, None)
            for fn_name, _ in self.registered_nodes[node_name]['as_fn_input']:
                output_mapping = self.registry[fn_name]['output_mapping']
                for new_node in output_mapping.values():
                    pop_nodes.append(new_node)


def register_agent(name, type, input_mapping, output_mapping, description, return_direct=False):
    def decorator(fn):
        AgentRegistry.register_cls_function(fn, name, type, input_mapping, output_mapping, description, return_direct)
        return fn
    return decorator
