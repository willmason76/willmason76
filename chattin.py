import os
import gradio as gr
import openai

openai.api_key = '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'

message_history = []

def chat(input, role="user"):
    message_history.append({"role": role, "content": f"{input}"})
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=message_history,
        temperature = 1.5,
        max_tokens = 200
    )
    reply_content = completion.choices[0].message.content
    print(reply_content)
    message_history.append({"role": "assistant", "content": f"{reply_content}"})
    response = [(message_history[i]['content'], message_history[i + 1]['content']) for i in range(2, len(message_history) - 1, 2)]
    return response

for i in range(2):
    user_input = input(" What do you want to know? ")
    print(chat(user_input))
    print()

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter").style(container=False)
    txt.submit(chat, txt, chatbot)

demo.launch()

#was running on http://127.0.0.1:7860
#if you want the website to be public for someone else's set `share=True` in launch()
