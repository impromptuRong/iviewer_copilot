import os
import json
import requests
import numpy as np
import pandas as pd
import gradio as gr

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from functools import lru_cache

from agents.utils.simpletiff import SimpleTiff
from agents.utils.utils_image import Slide
from tifffile import TiffFile

from llm_config import MODEL_REGISTRY, SYSTEM_PROMPT, RAG_PROMPT
from agents import AgentRegistry, RAGRouter, resize_pil
from fastapi import FastAPI, HTTPException, Request, Response, Depends
from fastapi.responses import JSONResponse
import subprocess


## Image Caption, LLM, RAG Registries
init_nodes = {'image', 'roi', 'mpp', 'core_nuclei_types', 'annotations', 'description',}
PATCH_SIZE = (512, 512)
OLLAMA_HOST_LLM = os.environ.get('OLLAMA_HOST_LLM', 'localhost')
OLLAMA_PORT_LLM = os.environ.get('OLLAMA_PORT_LLM', '11434')
## TODO: 
# 1. Implement MultimodalTextbox: https://www.gradio.app/docs/gradio/multimodaltextbox
# 2. Keep ChatHistory: https://www.gradio.app/guides/multimodal-chatbot-part1
# 3. Fix image download button: https://github.com/gradio-app/gradio/issues/6722
# 4. Use zoomable and draggable image container (plotly not working well, need new solution)


annotation_hostname = os.environ.get('ANNOTATION_HOST', 'localhost')
annotation_port = os.environ.get('ANNOTATION_PORT', '10020')
database_hostname = f"http://{annotation_hostname}:{annotation_port}"


@lru_cache(maxsize=8)
def _get_slide(slide_path):
    try:
        print(f"Excute remote slide: {slide_path}")
        if slide_path.startswith('http'):
            print(f"Use SimpleTiff")
            osr = SimpleTiff(slide_path)
            engine = 'simpletiff'
        else:
            print(f"Use TiffFile")
            osr = TiffFile(slide_path)
            engine = 'tifffile'
        slide = Slide(osr)
        slide.attach_reader(osr, engine=engine)

        return slide
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Failed to load slide from {slide_path}: {str(e)}")


def get_roi_tile(file, roi, image_size=(512, 512)):
    ## Pull ROI from slide
    slide = _get_slide(file)
    max_w, max_h = slide.level_dims[0]
    x0, y0, x1, y1 = roi
    # we add a padding of min(10, roi_h/w * 0.1) pixels
    pad_l = pad_r = min(10, (x1 - x0) * 0.1)
    pad_u = pad_d = min(10, (y1 - y0) * 0.1)
    x0, x1 = x0 - pad_l, x1 + pad_r
    y0, y1 = y0 - pad_u, y1 + pad_d
    # get ROI region
    x, y = max(x0, 0), max(y0, 0)
    w, h = min(x1, max_w) - x, min(y1, max_h) - y

    factor = min(w / image_size[0], h / image_size[1])
    if factor > 1.0:
        page = slide.get_resize_level(factor, downsample_only=True)
        scale_w, scale_h = slide.get_scales(page)[-1][0]
        x, y = int(round(x * scale_w)), int(round(y * scale_h))
        w, h = int(round(w * scale_w)), int(round(h * scale_h))
    else:
        page = 0
        x, y = int(round(x)), int(round(y))
        w, h = int(round(w)), int(round(h))

    patch = slide.get_patch([x, y, w, h], level=page)

    return patch


def get_roi_annotations(request_url, criteria={}):
    response = requests.post(request_url, json=criteria)
    if response.status_code == 200:
        resp = json.loads(response.content.decode('utf-8'))
        return resp
    else:
        return {}


def onload(request: gr.Request):
    request_args = {k.split('amp;')[-1]: v for k, v in request.query_params.items()}
    image_id, file = request_args['image_id'], request_args['file']
    x0, y0 = float(request_args['x0']), float(request_args['y0'])
    x1, y1 = float(request_args['x1']), float(request_args['y1'])
    patch = get_roi_tile(file, [x0, y0, x1, y1], PATCH_SIZE)
    # patch = go.Figure(go.Image(z=np.array(patch)))

    ## build agent_registry and rag_router
    agent_registry = AgentRegistry(init_nodes)
    agent_registry.update_cache('image', patch)  # add image patch
    agent_registry.update_cache('roi', [x0, y0, x1, y1])  # add roi coordinates
    agent_registry.update_cache('mpp', 0.25)  # add slide mpp
    agent_registry.update_cache('core_nuclei_types', ['tumor_nuclei', 'stromal_nuclei', 'immune_nuclei'])

    request_url = f"{database_hostname}/annotation/search?image_id={image_id}"
    # exclude 'description', otherwise will match db.description == description and return emtpy
    criteria = {k: v for k, v in request_args.items() if k != 'description'}  # 'annotator': ['yolov8-lung']} 'label': ['tumor', 'immune']}
    annotations = get_roi_annotations(request_url, criteria=criteria)
    agent_registry.update_cache('annotations', annotations)  # add annotations

    description = request_args.get('description')
    agent_registry.update_cache('description', [description])  # add descriptions

    rag_router = RAGRouter.from_agent_registry(
        agent_registry, 
        llm=request_args['rag'], 
        similarity_top_k=3,
        llm_cfgs={'temperature': 0, 'system_prompt': RAG_PROMPT},
    )

    agent_state = {
        'image_id': image_id,
        'roi': [x0, y0, x1, y1],
        'agent_registry': agent_registry, 
        'rag_router': rag_router,
    }
    print("***********************************")
    print(f"Registered Agents: ")
    print(agent_state['rag_router'].tools.keys())
    print("***********************************")

    return patch, [[None, description]], description or '', agent_state


def generate_comment(comment, agent_state, request: gr.Request):
    request_args = {k.split('amp;')[-1]: v for k, v in request.query_params.items()}
    x0, y0, x1, y1 = agent_state['roi']

    ## Init a message based on comment
    msg = comment or [[None, '']]
    if msg[0][-1]:
        yield msg, msg[0][-1]
    elif x1 - x0 < 100 or y1 - y0 < 100:
        # get region size, if it's too small, we don't auto generate messages
        yield msg, msg[0][-1]
    else:
        ## Run MLLM agents for image caption
        try:
            image_patch = agent_state['agent_registry'].cache['image']
            client = MODEL_REGISTRY.get_caption_model(request_args['caption'])
            print(f"Using client {request_args['caption']} (model={client.model})")
            image_size = client.configs.get('image_size', PATCH_SIZE)
            patch = resize_pil(image_patch, image_size)
            messages = client.build_messages(
                prompt=client.configs['init_prompts'], 
                system=client.configs['system'],
                images=[patch], 
                resize_image=True,
            )
            stream = client.stream(
                messages=messages,
                options=client.configs.get('options'),
            )

            for chunk in stream:
                # print(chunk, end='', flush=True)
                msg[0][-1] += chunk
                yield msg, msg[0][-1]
                
        except Exception as e:
            # msg[0][-1] += f"Failed to generate image captions with MLLM agents."
            import traceback
            error_msg = f"MLLM Error: {str(e)}\nTraceback: {traceback.format_exc()}"
            print(error_msg)
            msg[0][-1] += f"\n\nError generating captions: {str(e)}"
            yield msg, msg[0][-1]

        ## Run nuclei summary agents
        try:
            df = pd.DataFrame(agent_state['agent_registry'].cache['annotations'])
            if not df.empty:
                stats = df['label'].value_counts().to_dict()
                nuclei_summary = json.dumps(stats)
                msg[0][-1] += f"\n\nAdditionally, the following information are observed: "
                yield msg, msg[0][-1]
                msg[0][-1] += f"{nuclei_summary}."
                yield msg, msg[0][-1]
        except:
            msg[0][-1] += f"Failed to generate nuclei summary with agents."
            yield msg, msg[0][-1]


def user(user_message, history):
    if user_message:
        history += [[user_message, None]]

    return "", history


def update_description(comment, agent_state):
    description = comment[0][-1] if comment else ''
    if description != agent_state['agent_registry'].cache['description']:
        agent_state['agent_registry'].update_cache('description', [description])

    return agent_state


def llm_copilot(agent_state, history, request: gr.Request):
    if history and history[-1][-1] is None:
        prompt = history[-1][0]
        stream = agent_state['rag_router'].agent.stream_chat(message=prompt).response_gen

        history[-1][-1] = ""
        for chunk in stream:
            # print(chunk, end='', flush=True)
            history[-1][-1] += chunk
            yield history
    else:
        yield history


def clear_fn(agent_state):
    agent_state['rag_router'].agent.reset()
    return None


def toggle_image(state):
    btn_text = "Show image" if state else "Hide image"
    state = not state
    return gr.update(visible=state), state, gr.update(value=btn_text)


def toggle_comment(state):
    btn_text = "Show comment" if state else "Hide comment"
    state = not state
    return gr.update(visible=state), state, gr.update(value=btn_text)


def vote(data: gr.LikeData):
    if data.liked:
        print("You upvoted this response: " + data.value)
    else:
        print("You downvoted this response: " + data.value)


css = """
#copilot { height: 80vh; }
#image { flex-grow: 1; overflow: auto; }
#comment { flex-grow: 1; overflow: auto; height: 32vh; }
#chatbot { flex-grow: 1; overflow: auto; }
footer {visibility: hidden}
"""
# .button textarea {font-size: 12px !important}


with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="copilot"):
        agent_state = gr.State()
        with gr.Row():
            image = gr.ImageEditor(
                elem_id="image", 
                visible=False,
                height='32vh',
                show_download_button=True,
            )
            with gr.Tabs(elem_id='comment') as comment_tab:
                with gr.TabItem("comment"):
                    comment = gr.Chatbot(
                        elem_id="display", 
                        visible=True, 
                        height='32vh',
                        # label="Comment", 
                        show_label=False,
                        layout="panel", 
                        show_copy_button=True,
                    )
                with gr.TabItem("editor"):
                    editor = gr.Textbox(
                        elem_id="editor", 
                        visible=True, 
                        show_label=False,
                        show_copy_button=True,
                        lines=5,
                        max_lines=100,
                    )

        with gr.Row():
            image_state = gr.State(False)
            image_toggle_btn = gr.Button(value="Show image", elem_classes="button", size='sm', min_width=10)
            comment_state = gr.State(True)
            comment_toggle_btn = gr.Button(value="Hide comment", elem_classes="button", size='sm', min_width=10)
            regenerate_btn = gr.Button(value="Regenerate", elem_classes="button", size='sm', min_width=10)
            # mic = gr.Audio(source=["microphone"], type="filepath", streaming=True)

        chatbot = gr.Chatbot(
            elem_id="chatbot", label="Copilot", 
            show_copy_button=True,
            # show_share_button=True, 
        )

    with gr.Column(elem_id="prompts"):
        prompt = gr.Textbox(
            elem_id="promptbox", 
            placeholder="Message I-Viewer", 
            show_label=False,
        )
        with gr.Row():
            go_btn = gr.Button("Chat", elem_classes="button", size='sm', min_width=10)
            stop_btn = gr.Button("Stop", elem_classes="button", size='sm', min_width=10)
            clear_btn = gr.Button("Clear", elem_classes="button", size='sm', min_width=10)

    # Onload run generate_comment
    comment_fn = demo.load(onload, inputs=None, outputs=[image, comment, editor, agent_state]).success(
        generate_comment, inputs=[comment, agent_state], outputs=[comment, editor],
    )  #, _js=on_load)

    # Click regenerate button: stop onload run, clear comment content, regenerate comment
    comment_regnerate_fn = regenerate_btn.click(
        lambda: (None, ''), inputs=None, outputs=[comment, editor], cancels=[comment_fn], queue=False,
    ).success(
        generate_comment, inputs=[comment, agent_state], outputs=[comment, editor],
    )
    # sync markdown with editor when user edit the content
    editor.input(lambda x: [[None, x]] if x else None, inputs=[editor], outputs=[comment])

    # Click Enter after chat: ignore inputs if current chat is still running
    chat_submit_fn = prompt.submit(user, [prompt, chatbot], [prompt, chatbot]).then(
        update_description, inputs=[comment, agent_state], outputs=[agent_state],
    ).then(
        llm_copilot, inputs=[agent_state, chatbot], outputs=chatbot,
    )

    # Click Chat button: ignore inputs if current chat is still running
    chat_go_fn = go_btn.click(user, [prompt, chatbot], [prompt, chatbot]).then(
        update_description, inputs=[comment, agent_state], outputs=[agent_state],
    ).then(
        llm_copilot, inputs=[agent_state, chatbot], outputs=chatbot,
    )

    # Click Stop button: stop ongoging generation and chat
    stop_btn.click(None, None, None, cancels=[comment_fn, comment_regnerate_fn, chat_submit_fn, chat_go_fn], queue=False)

    # Click Clear button: stop ongoing chat, clear chatbot content
    clear_btn.click(clear_fn, [agent_state], [chatbot], cancels=[chat_submit_fn, chat_go_fn], queue=False)

    # Click Hide/Show Image/Comment button: hide/show comment content
    image_toggle_btn.click(toggle_image, [image_state], [image, image_state, image_toggle_btn])
    comment_toggle_btn.click(toggle_comment, [comment_state], [comment_tab, comment_state, comment_toggle_btn])

    chatbot.like(vote, None, None)


# demo.queue(default_concurrency_limit=10).launch(
#     server_name='0.0.0.0', server_port=10024, 
#     share=True, debug=True,
# )


app = FastAPI()
demo.queue(default_concurrency_limit=10)
app = gr.mount_gradio_app(app, demo, path="/copilot", root_path="/copilot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust the allowed origins as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/health")
async def check_health(request: Request):
    # return JSONResponse(content={"status": "healthy", "message": "Service is running!"})
    try:
        health_check = "OK"  
       # Get environment variables from Docker build args
        url = f"http://{OLLAMA_HOST_LLM}:{OLLAMA_PORT_LLM}"
        
        # Check connection using a POST request with payload
        curl_response = check_connection(url, method="GET")
        status_response = {
            "health": health_check,
            "environment": {
                "http_proxy": os.getenv("http_proxy"),
                "https_proxy": os.getenv("https_proxy"),
                "no_proxy": os.getenv("no_proxy"),
            },
            "details": f"Test connection to Ollama URL: {url}, Output: {curl_response['output']}"
        }
        return JSONResponse(content=status_response)
    except Exception as e:
         return JSONResponse(
            content={
                "health": "unhealthy",
                "details": f"Error: {str(e)}"
            },
            status_code=500  # Internal Server Error
        )

def check_connection(url:str, method: str = "GET", data: str = None):
    """Check the connection using curl"""
    try:
        curl_command = [
            'curl',
            '-X', method.upper(),
            url
        ]
        
        if data and method.upper() == "POST":
            curl_command.extend([
                "-H", "Content-Type: application/json",
                "-d", data
            ])     
        result= subprocess.run(curl_command, capture_output= True, text=True)
        if result.returncode == 0:
            return {"success": True, "output": result.stdout}
        else:
            return {"success": False, "output": result.stderr}
    
    except Exception as e:
        return {"success": False, "output": f"Error: {str(e)}"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9040)
