# I-Viewer Copilot for Digital Pathology
## Introduction
I-viewer copilot is a comprehensive online framework designed to address the needs of collaborative pathology analysis while harnessing the power of AI. I-viewer deployed advanced real-time AI-agents for different tasks and relies on MLLM for information integration and Human-AI collaboration. 

## Installation
1. Clone the repo
```
git clone https://github.com/impromptuRong/iviewer_copilot.git
cd iviewer_copilot
```
2. Create an `.env` file with the following contents to specify MLLM server.
```
SLIDES_DIR=input_slides_folder
DATABASE_DIR=./databases
HOSTNAME=machine_ip_address
OLLAMA_HOST=localhost
OLLAMA_PORT_CAPTION=11434
OLLAMA_PORT_CHATBOT=11435
OPENAI_API_KEY=openai_api_key
```
3. Start server with docker
```
docker compose up -d
```

Then you can connect the backend API with frontend server. Or view a demo by opening the `./templates/index.html`

