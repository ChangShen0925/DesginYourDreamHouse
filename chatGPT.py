import openai
openai.api_key = ""

def enhance_your_sentence(style, text, prompt):
    if len(style) == 0 or 'Other' in style:
        style = text
    PROMPT = " describe the overview of the outside of a house in the style of " + style + " which is " + prompt + "in 30 words"
    messages = [{"role": "system", "content": 
            PROMPT}]
    response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages)
    
    response_message = response["choices"][0]["message"].content
    response_message = response_message.replace('\n', ' ')
    response_message = response_message.replace('\\', '')
        
    return response_message

def enhance_your_sentence2(room, prompt):
    
    PROMPT = " describe the room of " + room + " which is " + prompt + " in 30 words"
    messages = [{"role": "system", "content": 
            PROMPT}]
    response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages)
    
    response_message = response["choices"][0]["message"].content
    response_message = response_message.replace('\n', ' ')
    response_message = response_message.replace('\\', '')
        
    return response_message
    
def enhance_your_sentence3(prompt):
    
    PROMPT = " describe the scenario of the music in the style of " + prompt + " in 15 words for example A ghostly choir chanting hauntingly beautiful hymns."
    messages = [{"role": "system", "content": 
            PROMPT}]
    response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages)
    
    response_message = response["choices"][0]["message"].content
    response_message = response_message.replace('\n', ' ')
    response_message = response_message.replace('\\', '')
        
    return response_message

def enhance_your_sentence4(prompt):
    
    PROMPT = prompt + ". Help me reshape the sentence and introduce to the audience in 20 words"
    messages = [{"role": "system", "content": 
            PROMPT}]
    response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages)
    
    response_message = response["choices"][0]["message"].content
    response_message = response_message.replace('\n', ' ')
    response_message = response_message.replace('\\', '')
        
    return response_message

def TranslateToChinese(prompt):
    
    PROMPT = prompt + "先翻译成中文，再必须必须至少润色到45个字以上，50个字以下"
    messages = [{"role": "system", "content": 
            PROMPT}]
    response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages)
    
    response_message = response["choices"][0]["message"].content
    response_message = response_message.replace('\n', ' ')
    response_message = response_message.replace('\\', '')
        
    return response_message