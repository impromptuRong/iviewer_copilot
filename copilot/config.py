## https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values
ollama_parameters = {
    'num_ctx': None,  # Sets the size of the context window used to generate the next token. (Default: 2048)
    'num_predict': None,  # Maximum number of tokens to predict when generating text. (Default: 128, -1 = infinite generation, -2 = fill context)
    'temperature': None,  # The temperature of the model. Increasing the temperature will make the model answer more creatively. (Default: 0.8)
    'top_k': None,  # Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative. (Default: 40)
    'top_p': None,  # Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. (Default: 0.9)
    'repeat_last_n': None,  # Sets how far back for the model to look back to prevent repetition. (Default: 64, 0 = disabled, -1 = num_ctx)
    'repeat_penalty': None,  # Sets how strongly to penalize repetitions. A higher value (e.g., 1.5) will penalize repetitions more strongly, while a lower value (e.g., 0.9) will be more lenient. (Default: 1.1)
    'seed': None,  # Sets the random number seed to use for generation. Setting this to a specific number will make the model generate the same text for the same prompt. (Default: 0)
}


llava = {
    'model': 'llava:13b-v1.6-vicuna-q8_0',
    'system': 'You possess comprehensive training in surgical pathology and possess an adept understanding of the visual elements present in Hematoxylin and Eosin (H&E) images. Utilizing histological terminology, analyze the histology features, patterns, and nuclei morphologies depicted in the images. Provide a comprehensive summary addressing the following aspects based on the content of the images: ',
    'init_prompts': '1. Interpret the tumor interface with the surrounding normal tissue, delineating between circumscribed, encapsulated, infiltrative, lobular, and pushing borders.\n2. Describe the growth patterns or secondary structures/architectural patterns observed, including acinar, alveolar, cribriform, glandular, hobnailed, lepidic, micropapillary, nodular, papillary, trabecular, and solid growth.\n3. Identify and characterize cellular/nuclear/cytoplasmic morphological features, focusing on nuclear shape and size, nuclear pleomorphism, nuclear-to-cytoplasmic ratio, and the presence of prominent nucleoli.',
    'image_size': (224, 224),
    'options': ollama_parameters,
}


llama3 = {
    'model': 'llama3:8b-instruct-fp16',
    'system': 'You have complete training in pathology and histology and you are capable to generate pathological descriptions, summarize information, infer diagnosis. Answer user questions based on the following facts observed: ',
    'options': ollama_parameters,
}


# https://platform.openai.com/docs/api-reference/chat/create
gpt_parameters = {
    # 'max_tokens': None,  # The maximum number of tokens that can be generated in the chat completion.
    'temperature': None,  # What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. Defaults to 1. We generally recommend altering this or top_p but not both.
    'top_p': None,  # An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered. Defaults to 1. We generally recommend altering this or temperature but not both.    
    'frequency_penalty': None,  # Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim. Defaults to 0
    'presence_penalty': None,  # Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics. Defaults to 0
    'seed': None,
}


gpt4v = {
    'model': 'gpt-4-turbo',
    'system': 'You possess comprehensive training in surgical pathology and possess an adept understanding of the visual elements present in Hematoxylin and Eosin (H&E) images. Utilizing histological terminology, analyze the histology features, patterns, and nuclei morphologies depicted in the images. Provide a comprehensive summary addressing the following aspects based on the content of the images: ',
    'init_prompts': '1. Interpret the tumor interface with the surrounding normal tissue, delineating between circumscribed, encapsulated, infiltrative, lobular, and pushing borders.\n2. Describe the growth patterns or secondary structures/architectural patterns observed, including acinar, alveolar, cribriform, glandular, hobnailed, lepidic, micropapillary, nodular, papillary, trabecular, and solid growth.\n3. Identify and characterize cellular/nuclear/cytoplasmic morphological features, focusing on nuclear shape and size, nuclear pleomorphism, nuclear-to-cytoplasmic ratio, and the presence of prominent nucleoli.',
    'image_size': (512, 512),
    'options': gpt_parameters,
}

gpt4o = {
    'model': 'gpt-4o',
    'system': 'You possess comprehensive training in surgical pathology and possess an adept understanding of the visual elements present in Hematoxylin and Eosin (H&E) images. Utilizing histological terminology, analyze the histology features, patterns, and nuclei morphologies depicted in the images. Provide a comprehensive summary addressing the following aspects based on the content of the images: ',
    'init_prompts': '1. Interpret the tumor interface with the surrounding normal tissue, delineating between circumscribed, encapsulated, infiltrative, lobular, and pushing borders.\n2. Describe the growth patterns or secondary structures/architectural patterns observed, including acinar, alveolar, cribriform, glandular, hobnailed, lepidic, micropapillary, nodular, papillary, trabecular, and solid growth.\n3. Identify and characterize cellular/nuclear/cytoplasmic morphological features, focusing on nuclear shape and size, nuclear pleomorphism, nuclear-to-cytoplasmic ratio, and the presence of prominent nucleoli.',
    'image_size': (512, 512),
    'options': gpt_parameters,
}


gpt4o_chat = {
    'model': 'gpt-4o',
    'system': 'You possess comprehensive training in surgical pathology and possess an adept understanding of the visual elements present in Hematoxylin and Eosin (H&E) images. Utilizing histological terminology, analyze the histology features, patterns, and nuclei morphologies depicted in the images. Provide answers to users based on the following facts and medical knowledge: ',
    'options': gpt_parameters,
}

gpt35_chat = {
    'model': 'gpt-3.5-turbo',
    'system': 'You possess comprehensive training in surgical pathology and possess an adept understanding of the visual elements present in Hematoxylin and Eosin (H&E) images. Utilizing histological terminology, analyze the histology features, patterns, and nuclei morphologies depicted in the images. Provide answers to users based on the following facts and medical knowledge: ',
    'options': gpt_parameters,
}
