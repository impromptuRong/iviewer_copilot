import base64
import openai
import ollama

from PIL import Image
from io import BytesIO

def resize_pil(img, output_size, keep_ratio=True, resample=Image.BILINEAR):
    w0, h0 = img.size
    if w0 == output_size[0] and h0 == output_size[1]:
        return img
    if keep_ratio:
        factor = min(output_size[0] / w0, output_size[1] / h0)
        output_size = (int(w0 * factor), int(h0 * factor))

    return img.resize(output_size, resample=resample)

def pil2bytes(img, format=None, default_format='PNG'):
    format = format or img.format or default_format
    buff = BytesIO()
    img.save(buff, format=format)
    image_bytes = buff.getvalue()
    # print(f"image size: {sys.getsizeof(image_bytes)/1048576} mb.")

    return image_bytes, format

def bytes2pil(img_bytes):
    return Image.open(BytesIO(img_bytes))


class OllamaClient:
    def __init__(self, configs, host='http://localhost:11434'):
        self.client = ollama.Client(host=host)
        # print(self.client.list())
        self.configs = configs
        self.load_model(configs)

    def load_model(self, configs):
        self.model = configs.get('model', 'llama3')
        # messages = [{'role': 'user', 'content': 'say hi'}]
        options = self.configs.get('options', {})
        try:
            self.client.pull(self.model)
            output = self.chat(messages=None, options=options)
            print(f"Load Successfully: model={self.model}, options={options}. \n{output}")
            return True
        except Exception as e:  # ollama.ResponseError as e:
            print('Error:', e)
            return False

    def stream(self, messages, options={}):
        print(options)
        stream = self.client.chat(
            model=self.model, 
            messages=messages, 
            stream=True,
            options=options,
        )

        for chunk in stream:
            fragment = chunk['message']['content']
            # print(fragment, end='', flush=True)
            yield fragment

    def chat(self, messages, options={}):
        response = self.client.chat(
            model=self.model, 
            messages=messages, 
            stream=False,
            options=options,
        )

        return response['message']['content']

    def build_messages(self, prompt, system=None, history=None, 
                       images=None, resize_image=False, **kwargs):
        messages = []
        system = system or self.configs.get('system')
        if system is not None:
            messages.append({
                'role': 'system',
                'content': system, 
            })

        if history:
            for user_info, agent_info in history:
                messages.append({'role': 'user', 'content': user_info})
                messages.append({'role': 'assistant', 'content': agent_info})

        user_content = {
            'role': 'user',
            'content': prompt,
        }
        if images:
            images_encoded = []
            image_size = self.configs.get('image_size')
            for img in images:
                if resize_image and image_size:
                    img = resize_pil(img, image_size, keep_ratio=True)
                img_bytes, _ = pil2bytes(img, format='jpeg')
                img_encoded = base64.b64encode(img_bytes).decode('utf-8')
                images_encoded.append(img_encoded)

            user_content['images'] = images_encoded

        messages.append(user_content)

        return messages


class GPTClient:
    def __init__(self, configs):
        self.model = configs.get('model', 'gpt-4o')
        self.configs = configs
        self.client = openai.OpenAI()
        #     organization=configs['organization'],
        #     project=configs['project'],
        #     api_key=configs['api_key'],
        # )

    def stream(self, messages, options={}):
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            **options,
        )

        for chunk in stream:
            fragment = chunk.choices[0].delta.content
            if fragment is not None:
                # print(fragment, end='', flush=True)
                yield fragment

    def chat(self, messages, options={}):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=False,
            **options,
        )

        return response.choices[0].message.content
    
    def build_messages(self, prompt, system=None, history=None, 
                       images=None, resize_image=False, **kwargs):
        messages = []
        system = system or self.configs.get('system')
        if system is not None:
            messages.append({
                'role': 'system',
                'content': system, 
            })

        if history:
            for user_info, agent_info in history:
                messages.append({'role': 'user', 'content': user_info})
                messages.append({'role': 'assistant', 'content': agent_info})

        user_content = [
            {"type": "text", "text": prompt,},
        ]
        if images:
            image_size = self.configs.get('image_size')
            for img in images:
                if resize_image and image_size:
                    img = resize_pil(img, image_size, keep_ratio=True)
                img_bytes, image_type = pil2bytes(img, format='jpeg')
                img_encoded = base64.b64encode(img_bytes)                
                user_content.append({"type": "image_url", "image_url": {"url": f"data:image/{image_type};base64,{img_encoded.decode('utf-8')}",}})

        messages.append({"role": "user", "content": user_content,})

        return messages
