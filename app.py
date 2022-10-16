import gradio as gr
import torch
import os
from diffusers import StableDiffusionPipeline

def dummy(images, **kwargs):
    return images, False

model_id = "CompVis/stable-diffusion-v1-4"

AUTH_TOKEN = os.environ.get('AUTH_TOKEN')
if not AUTH_TOKEN:
    with open('/root/.huggingface/token') as f:
        lines = f.readlines()
        AUTH_TOKEN = lines[0]

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print('Nvidia GPU detected!')
    share = True
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        use_auth_token=AUTH_TOKEN,
        revision="fp16",
        torch_dtype=torch.float16
    )
else:
    print('No Nvidia GPU in system!')
    share = False
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        use_auth_token=AUTH_TOKEN
    )

pipe.to(device)
pipe.safety_checker = dummy

def infer(prompt="", samples=4, steps=20, scale=7.5, seed=1437181781):
    generator = torch.Generator(device=device).manual_seed(seed)
    images = []
    images_list = pipe(
        [prompt] * samples,
        num_inference_steps=steps,
        guidance_scale=scale,
        generator=generator,
    )

    for i, image in enumerate(images_list["sample"]):
        images.append(image)

    return images

gr.Interface(
    fn=infer,
    inputs=[
        gr.Textbox(lines=2, placeholder="Prompt here..."),
        gr.Slider(label="Images", minimum=1, maximum=4, value=4, step=1),
        gr.Slider(label="Steps", minimum=1, maximum=50, value=20, step=1),
        gr.Slider(label="Guidance Scale", minimum=0, maximum=50, value=7.5, step=0.1),
        gr.Slider(label="Seed", minimum=0, maximum=2147483647, step=1, randomize=True)
    ],
    outputs=gr.Gallery(
        label="Generated images",
        show_label=False,
        elem_id="gallery"
    ).style(grid=[2], height="auto"),
).launch().launch(
    share=share,
    enable_queue=True
)
