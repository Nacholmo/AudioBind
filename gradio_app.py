import os
import gradio as gr
import torch
from languagebind import LanguageBind, transform_dict, LanguageBindImageTokenizer, to_device
import sounddevice as sd
import wavio

audio_dir_input = "/home/sundae/Documentos/audio"
image_dir_input = "/home/sundae/images"
text_file_input = "/home/sundae/tags.txt"  # Replace with defaults paths
recording = False
samplerate = 44100
duration = 2

def record_and_update_audio(image_dir, duration):
    global recording
    if recording:
        filename = os.path.join(audio_dir_input, "mic.wav")
        record_audio(filename, duration, samplerate)
        similarity_text, top_images = audio_to_images(filename, image_dir)
        return similarity_text, top_images
    else:
        return "Recording stopped", []
    

def toggle_recording(image_dir):
    global recording
    recording = not recording
    if recording:
        return "Recording started...", [], gr.Button("Stop Recording")
    else:
        return "Recording stopped", [], gr.Button("Start Recording")

def load_text_lines(text_file):
    with open(text_file, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]

def audio_to_texts(audio, text_lines):
    audio_input = to_device(modality_transform['audio'](audio), device)
    with torch.no_grad():
        audio_embedding = model({'audio': audio_input})['audio']

    leaderboard = []
    for i, text in enumerate(text_lines):
        text_input = to_device(modality_transform['language'](text, max_length=77, padding='max_length',
                                                                  truncation=True, return_tensors='pt'), device)
        with torch.no_grad():
            text_embedding = model({'language': text_input})['language']
        similarity = (audio_embedding @ text_embedding.T).item()
        leaderboard.append((i, text, similarity))
    leaderboard.sort(key=lambda x: x[2], reverse=True)
    leaderboard_text = "\n".join(f"{i+1}. {text}: {similarity:.4f}" for i, text, similarity in leaderboard[:5])
    return leaderboard_text

def record_and_update_text(text_file, duration):
    global recording
    if recording:
        filename = os.path.join(audio_dir_input, "mic.wav")
        record_audio(filename, duration, samplerate)
        text_lines = load_text_lines(text_file)
        similarity_text = audio_to_texts(filename, text_lines)
        return similarity_text
    else:
        return "Recording stopped"
    
def continuous_audio_recording(audio_dir, update_callback):
    samplerate = 44100
    duration = 2
    global recording
    while (recording):
        filename = os.path.join(audio_dir, "mic.wav")
        record_audio(filename, duration, samplerate)
        print(f"Saved {filename}")
        update_callback()


def stop_continuous_audio_recording():
    global recording_thread
    recording_thread.join()
    recording = False
    print("Stopped recording")


def calculate_similarity_of_last_audio():
    similarity_text, top_images = audio_to_images_last(audio_dir_input, image_dir_input)
    print(f"similarity_text = {similarity_text}")
    
    return similarity_text, top_images

def update_top_continuous_images():
    similarity_text, top_images = audio_to_images_last(audio_dir_input, image_dir_input)
    top_continuous_images.value = top_images

    return similarity_text

def record_audio(filename, duration, samplerate):
    print("Recording...")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    print("Saving...")
    wavio.write(filename, recording, rate=samplerate, sampwidth=2)

def audio_to_images_last(audio_dir, image_dir):
    latest_audio_file = os.path.join(audio_dir, "mic.wav")
    print(f"Latest audio file: {latest_audio_file}")

    similarity_text, top_images = audio_to_images(latest_audio_file, image_dir)
    return similarity_text, top_images


def audio_to_language(audio, language):
    inputs = {}
    inputs['audio'] = to_device(modality_transform['audio'](audio), device)
    inputs['language'] = to_device(modality_transform['language'](language, max_length=77, padding='max_length',
                                                                  truncation=True, return_tensors='pt'), device)
    with torch.no_grad():
        embeddings = model(inputs)
    return (embeddings['audio'] @ embeddings['language'].T).item()


def audio_to_image(audio, image):
    print(f"Calculating similarity of audio {audio} to image {image}")
    inputs = {}
    inputs['audio'] = to_device(modality_transform['audio'](audio), device)
    inputs['image'] = to_device(modality_transform['image'](image), device)
    with torch.no_grad():
        embeddings = model(inputs)
    similarity = (embeddings['audio'] @ embeddings['image'].T).item()
    print(f"Similarity: {similarity}")
    return similarity

def audio_to_images(audio, image_dir):
    global image_embeddings, current_image_dir
    if 'current_image_dir' not in globals():
        current_image_dir = None

    if image_dir != current_image_dir:
        print(f"Loading images from directory: {image_dir}")
        image_embeddings = {}
        for image_file in os.listdir(image_dir):
            if image_file.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(image_dir, image_file)
                image_input = to_device(modality_transform['image'](image_path), device)
                with torch.no_grad():
                    image_embedding = model({'image': image_input})['image']
                image_embeddings[image_file] = image_embedding
        current_image_dir = image_dir
        print(f"Loaded {len(image_embeddings)} images")

    print(f"Calculating similarity of audio {audio} to images in directory {image_dir}")
    audio_input = to_device(modality_transform['audio'](audio), device)
    with torch.no_grad():
        audio_embedding = model({'audio': audio_input})['audio']

    leaderboard = []
    for image_file, image_embedding in image_embeddings.items():
        similarity = (audio_embedding @ image_embedding.T).item()
        leaderboard.append((image_file, similarity))
    leaderboard.sort(key=lambda x: x[1], reverse=True)
    leaderboard_text = "\n".join(f"{name}: {similarity}" for name, similarity in leaderboard[:5])
    top_images = [os.path.join(image_dir, name) for name, _ in leaderboard[:4]]
    return leaderboard_text, top_images

if __name__ == '__main__':
    device = 'cuda:0'
    device = torch.device(device)
    clip_type = {
        'audio': 'LanguageBind_Audio_FT',  # also LanguageBind_Audio
        'image': 'LanguageBind_Image',
    }

    print("Loading LanguageBind model...")
    model = LanguageBind(clip_type=clip_type, use_temp=False)
    model = model.to(device)
    model.eval()
    pretrained_ckpt = f'LanguageBind/LanguageBind_Audio_FT'
    tokenizer = LanguageBindImageTokenizer.from_pretrained(pretrained_ckpt, cache_dir='./cache_dir/tokenizer_cache_dir')
    modality_transform = {c: transform_dict[c](model.modality_config[c]) for c in clip_type}
    modality_transform['language'] = tokenizer
    print("LanguageBind model loaded")
    image_embeddings = {}
    current_image_dir = None

with gr.Blocks(title="AudioStuff 🐈") as demo:
    with gr.Tabs():
        with gr.TabItem("Similarity of Audio to Text"):
            audio = gr.Audio(type="filepath", label='Audio Input')
            language_a = gr.Textbox(lines=2, label='Text Input')
            out_a = gr.Textbox(label='Similarity of Audio to Text')
            b_a = gr.Button("Calculate similarity of Audio to Text")

            b_a.click(
                audio_to_language,
                inputs=[audio, language_a],
                outputs=out_a
                )
        
        with gr.TabItem("Similarity of Audio to Image"):
            audio_i = gr.Audio(type="filepath", label='Audio Input')
            image_i = gr.Image(type="filepath", label='Image Input')
            out_i = gr.Textbox(label='Similarity of Audio to Image')
            b_i = gr.Button("Calculate similarity of Audio to Image")

            b_i.click(
                audio_to_image,
                inputs=[audio_i, image_i],
                outputs=out_i
                )
        
        with gr.TabItem("Similarity of Audio to Images"):
            audios = gr.Audio(type="filepath", label='Audio Input')
            image_dir = gr.Textbox(lines=1, label='Image Directory')
            out_is = gr.Textbox(label='Similarity of Audio to Images')
            top_images = gr.Gallery(label='Top Images')
            b_is = gr.Button("Calculate similarity of Audio to Images")

            b_is.click(
                audio_to_images,
                inputs=[audios, image_dir],
                outputs=[out_is, top_images]
                )
            
        with gr.TabItem("Continuous Audio Similarity"):
            image_dir = gr.Textbox(lines=1, label='Image Directory', value=image_dir_input)
            duration_slider = gr.Slider(minimum=1, maximum=10, step=1, label='Duration (seconds)', value=duration)
            out_continuous_similarity = gr.Textbox(label='Similarity of Audio to Images')
            top_continuous_images = gr.Gallery(label='Top Images')
            record_button = gr.Button("Start Recording")
            record_button.click(
                toggle_recording,
                inputs=[image_dir],
                outputs=[out_continuous_similarity, top_continuous_images, record_button]
            )
            demo.load(
                record_and_update_audio,
                inputs=[image_dir, duration_slider],
                outputs=[out_continuous_similarity, top_continuous_images],
                every=duration
            )
        
        with gr.TabItem("Continuous Text Similarity"):
            text_file = gr.Textbox(lines=1, label='Text File', value=text_file_input)
            duration_slider_text = gr.Slider(minimum=1, maximum=10, step=1, label='Duration (seconds)', value=duration)
            out_continuous_text_similarity = gr.Textbox(label='Similarity of Audio to Texts')
            record_button_text = gr.Button("Start Recording")
            record_button_text.click(
                toggle_recording,
                inputs=[text_file],
                outputs=[out_continuous_text_similarity, record_button_text]
            )
            demo.load(
                record_and_update_text,
                inputs=[text_file, duration_slider_text],
                outputs=[out_continuous_text_similarity],
                every=duration
            )


    print("Starting Gradio demo...")
    demo.launch()
