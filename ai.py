import os
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import requests
from pyngrok import ngrok
from flask import Flask, request, jsonify
import io
from google.colab import drive
from pathlib import Path

# Kết nối Google Drive
drive.mount('/content/drive')

# Đường dẫn đến thư mục lưu model trên Google Drive
MODEL_PATH = '/content/drive/MyDrive/AI_Models/Vintern-1B-v2'

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0] 
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    if isinstance(image_file, bytes):
        image = Image.open(io.BytesIO(image_file)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

app = Flask(__name__)

# Kiểm tra và tải model từ Google Drive nếu có
model_path = Path(MODEL_PATH)
if model_path.exists():
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=False)
else:
    # Nếu chưa có model trong Drive, tải về và lưu vào Drive
    model_name = "5CD-AI/Vintern-1B-v2"
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    
    # Lưu model và tokenizer vào Google Drive
    os.makedirs(MODEL_PATH, exist_ok=True)
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)

@app.route('/')
def home():
    return 'Hello World!'

@app.route('/get', methods=['POST']) 
def process_image():
    # Xử lý trường hợp có ảnh
    if 'image' in request.files:
        image_file = request.files['image'].read()  # Đọc file dưới dạng bytes
        question = request.form.get('question', '<image>\nMô tả hình ảnh một cách chi tiết.')
        
        pixel_values = load_image(image_file, max_num=6).to(torch.bfloat16).cuda()
        generation_config = dict(max_new_tokens=512, do_sample=False, num_beams=3, repetition_penalty=3.5)
        
        response = model.chat(tokenizer, pixel_values, question, generation_config)
        
        return jsonify({
            'question': question,
            'response': response
        })
    
    # Xử lý trường hợp chỉ có câu hỏi
    else:
        question = request.form.get('question')
        if not question:
            return jsonify({'error': 'No question provided'}), 400
            
        generation_config = dict(max_new_tokens=512, do_sample=False, num_beams=3, repetition_penalty=3.5)
        response = model.chat(tokenizer, None, question, generation_config)
        
        return jsonify({
            'question': question,
            'response': response
        })

if __name__ == '__main__':
    # Khởi tạo ngrok token
    ngrok.set_auth_token("Token")
    
    # Khởi tạo ngrok tunnel
    ngrok_tunnel = ngrok.connect(5000)
    print('Ngrok tunnel URL:', ngrok_tunnel.public_url)
    
    # Chạy Flask app 
    app.run(port=5000)