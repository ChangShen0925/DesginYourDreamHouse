import gradio as gr
with gr.Blocks() as demo:
    
    gr.HTML("""
                <html>
                <head>
                </head>
                <body>
                <img src="https://www.yephome.com.au/assets/image/logo/Yep-logo-white-home.svg" width="219" height="33"/>   

                </body>
                </html>
                """)

if __name__ == "__main__":
    demo.queue().launch(share=True, debug=True)