# I-Viewer Copilot for Digital Pathology
## Introduction
I-viewer copilot is a comprehensive online framework designed to address the needs of collaborative pathology analysis while harnessing the power of AI. I-viewer deployed advanced real-time AI-agents for different tasks and relies on MLLM for information integration and Human-AI collaboration. 

## Installation
1. Clone the repo
```
git clone https://github.com/impromptuRong/iviewer_copilot.git
cd iviewer_copilot
```

2. Create an `.env` file with the following contents to specify MLLM service.
```
SLIDES_DIR=input_slides_folder
DATABASE_DIR=./databases
OLLAMA_HOST=localhost
OLLAMA_PORT_CAPTION=11434
OLLAMA_PORT_CHATBOT=11435
OPENAI_API_KEY=openai_api_key
```
I-Viewer rely on `ollama` or `openai API` to enable chatbot and captioning service. For GPT user, put openai_api_key into the `.env` file as above. For `ollama` user, we recommend to host ollama service on a separate server. (Instructions about how to set up ollama and LLM models can be found here: https://github.com/ollama/ollama)

3. Start server with docker
```
docker compose up -d
```

Then you can connect the backend API with frontend server. Or view a demo by opening the `./templates/index.html`

## Extend I-Viewer with customer analysis pipeline
User can add their own pipeline into I-Viewer with `offline` interface and `online` interface. Basically it takes three steps:
```
## Create a generator agent
class GeneratorAgent:
    def prepare_inputs(self, requests):
        request_params = decode(requests)
        boxes = get_bbox(request_params)
        roi_image = get_roi_tile(request_params)
        ...
        
        return {'roi_image': roi_image, 'boxes': boxes, ...}
    
    def analysis_offline(self, inputs):
        serialized_item = serialize(inputs)
        await redis_client.stream_push(registry, serialized_item)
    
    def analysis_online(self, inputs):
        outputs = agents.analysis_online(inputs)
        return Response(outputs)

## Create a analysis agent
class AnalysisAgent:
    def predict(self, inputs):
        return pipeline(inputs)
    
    def analysis_offline(self):
        serialized_item = redis_client.stream_fetch(registry)
        inputs = deserilize(serialized_item)
        outputs = self.predict(inputs)
        export_to_db(postprocess(outputs))

    def analysis_online(self, inputs):
        outputs = self.predict(inputs)
        return postprocess(outputs)

## Register model and generator
MODEL_REGISTRY = ModelRegistry()
MODEL_REGISTRY.register("registry_name", "model", AnalysisAgent)
MODEL_REGISTRY.register("registry_name", "generator", GeneratorAgent)
```
