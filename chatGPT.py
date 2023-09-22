import openai
openai.api_key = "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"

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
    
    PROMPT = "This is the audio data read by AudioSegment.from_file by using audio.get_array_of_samples(), help me extend it to at least 1 minutes" + prompt 
    messages = [{"role": "system", "content": 
            PROMPT}]
    response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages)
    
    response_message = response["choices"][0]["message"].content
    response_message = response_message.replace('\n', ' ')
    response_message = response_message.replace('\\', '')
        
    return response_message