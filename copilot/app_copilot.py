import os
import json
import time
import requests
import numpy as np
import pandas as pd
import gradio as gr

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from functools import lru_cache

from utils.simpletiff import SimpleTiff
from utils.utils_image import Slide
from model_registry import MODEL_REGISTRY

## TODO: 
# 1. Implement MultimodalTextbox: https://www.gradio.app/docs/gradio/multimodaltextbox
# 2. Keep ChatHistory: https://www.gradio.app/guides/multimodal-chatbot-part1
# 3. Fix image download button: https://github.com/gradio-app/gradio/issues/6722
# 4. Use zoomable and draggable image container (plotly not working well, need new solution)


annotation_hostname = os.environ.get('ANNOTATION_HOST', 'localhost')
annotation_port = os.environ.get('ANNOTATION_PORT', '9020')
database_hostname = f"http://{annotation_hostname}:{annotation_port}"


@lru_cache(maxsize=8)
def _get_slide(slide_path):
    try:
        print(f"Excute remote slide: {slide_path}")
        osr = SimpleTiff(slide_path)
        slide = Slide(osr)
        slide.attach_reader(osr, engine='simpletiff')

        return slide
    except:
        raise HTTPException(status_code=404, detail=f"{slide_path} not found")


def get_roi_tile(request: gr.Request, image_size=(512, 512)):
    request_args = {k.split('amp;')[-1]: v for k, v in request.query_params.items()}

    ## Pull ROI from slide
    slide = _get_slide(request_args['file'])
    max_w, max_h = slide.level_dims[0]
    x0, y0 = float(request_args['x0']), float(request_args['y0'])
    x1, y1 = float(request_args['x1']), float(request_args['y1'])
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


def get_nuclei_stats(request: gr.Request):
    request_args = {k.split('amp;')[-1]: v for k, v in request.query_params.items()}
    image_id, registry = request_args['image_id'], request_args['registry']

    request_url = f"{database_hostname}/annotation/search?image_id={image_id}"
    criteria = {**request_args, 'annotator': ['yolov8-lung']}  # 'label': ['tumor', 'immune']}
    response = requests.post(request_url, json=criteria)
    if response.status_code == 200:
        resp = json.loads(response.content.decode('utf-8'))
        df = pd.DataFrame(resp)
        if not df.empty:
            return df['label'].value_counts().to_dict()
        else:
            return {}
    else:
        return {}


def onload(request: gr.Request):
    request_args = {k.split('amp;')[-1]: v for k, v in request.query_params.items()}

    client = MODEL_REGISTRY.get_caption_model(request_args['registry'])
    print(f"Using client {request_args['registry']} (model={client.model})")
    image_size = client.configs.get('image_size', (512, 512))
    patch = get_roi_tile(request, image_size)
    # patch = go.Figure(go.Image(z=np.array(patch)))

    msg = [[None, request_args.get('description')]]
    return patch, msg


def generate_comment(comment, request: gr.Request):
    request_args = {k.split('amp;')[-1]: v for k, v in request.query_params.items()}
    
    x0, y0 = float(request_args['x0']), float(request_args['y0'])
    x1, y1 = float(request_args['x1']), float(request_args['y1'])

    ## Init a message based on comment
    msg = comment or [[None, '']]
    if msg[0][-1]:
        yield msg
    elif x1 - x0 < 100 or y1 - y0 < 100:
        # get region size, if it's too small, we don't auto generate messages
        yield msg
    else:
        ## Run MLLM agents for image caption
        try:
            client = MODEL_REGISTRY.get_caption_model(request_args['registry'])
            print(f"Using client {request_args['registry']} (model={client.model})")
            image_size = client.configs.get('image_size', (512, 512))
            patch = get_roi_tile(request, image_size)
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
                yield msg
        except:
            msg[0][-1] += f"Failed to generate image captions with MLLM agents."
            yield msg

        ## Run nuclei summary agents
        try:
            stats = get_nuclei_stats(request)
            if stats:
                nuclei_summary = json.dumps(stats)
                msg[0][-1] += f"\n\nAdditionally, the following information are observed: "
                yield msg
                msg[0][-1] += f"{nuclei_summary}."
                yield msg
        except:
            msg[0][-1] += f"Failed to generate nuclei summary with agents."
            yield msg


def user(user_message, history):
    if user_message:
        history += [[user_message, None]]

    return "", history


def llm_copilot(comment, history, request: gr.Request):
    if history and history[-1][-1] is None:
        ## Construct messages
        request_args = {k.split('amp;')[-1]: v for k, v in request.query_params.items()}

        client = MODEL_REGISTRY.get_chatbot_model(request_args['registry'])
        print(f"Using client {request_args['registry']} (model={client.model})")

        # build history and messages
        description = comment[0][-1] if comment else ''
        messages = client.build_messages(
            prompt=history[-1][0], 
            system=client.configs['system'] + description,
            history=history[:-1],
        )

        stream = client.stream(
            messages=messages,
            options=client.configs.get('options'),
        )

        history[-1][-1] = ""
        for chunk in stream:
            # print(chunk, end='', flush=True)
            history[-1][-1] += chunk
            yield history
    else:
        yield history


def clear_fn():
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
#comment { flex-grow: 1; overflow: auto; }
#chatbot { flex-grow: 1; overflow: auto; }
footer {visibility: hidden}
"""
# .button textarea {font-size: 12px !important}


with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="copilot"):
        with gr.Row():
            image = gr.ImageEditor(
                elem_id="image", 
                visible=False,
                height='32vh',
                show_download_button=True,
            )
            comment = gr.Chatbot(
                elem_id="comment", 
                visible=True, 
                height='32vh',
                label="Comment", 
                layout="panel", 
                show_copy_button=True,
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
    comment_fn = demo.load(onload, inputs=None, outputs=[image, comment]).then(
        generate_comment, inputs=[comment], outputs=[comment],
    )  #, _js=on_load)

    # Click regenerate button: stop onload run, clear comment content, regenerate comment
    comment_regnerate_fn = regenerate_btn.click(
        clear_fn, inputs=None, outputs=[comment], cancels=[comment_fn],
    ).then(
        generate_comment, inputs=[comment], outputs=[comment],
    )

    # Click Enter after chat: ignore inputs if current chat is still running
    chat_submit_fn = prompt.submit(user, [prompt, chatbot], [prompt, chatbot]).then(
        llm_copilot, [comment, chatbot], chatbot
    )
    # chat = prompt.submit(llm_copilot, [prompt, comment, chatbot], [prompt, chatbot])

    # Click Chat button: ignore inputs if current chat is still running
    chat_go_fn = go_btn.click(user, [prompt, chatbot], [prompt, chatbot]).then(
        llm_copilot, [comment, chatbot], chatbot
    )

    # Click Stop button: stop ongoging generation and chat
    stop_btn.click(None, None, None, cancels=[comment_fn, comment_regnerate_fn, chat_submit_fn, chat_go_fn])

    # Click Clear button: stop ongoing chat, clear chatbot content
    clear_btn.click(clear_fn, None, [chatbot], cancels=[chat_submit_fn, chat_go_fn])

    # Click Hide/Show Image/Comment button: hide/show comment content
    image_toggle_btn.click(toggle_image, [image_state], [image, image_state, image_toggle_btn])
    comment_toggle_btn.click(toggle_comment, [comment_state], [comment, comment_state, comment_toggle_btn])

    chatbot.like(vote, None, None)


# demo.queue(default_concurrency_limit=10).launch(
#     server_name='0.0.0.0', server_port=10024, 
#     share=True, debug=True,
# )


app = FastAPI()
demo.queue(default_concurrency_limit=10)
app = gr.mount_gradio_app(app, demo, path="/copilot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust the allowed origins as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9040)
