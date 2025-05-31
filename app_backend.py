import os
import glob
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoFeatureExtractor, AutoModel # Import AutoFeatureExtractor, AutoModel
import torch
import face_recognition # For face detection (AI)
import textwrap # For neatly wrapping LLM output

from flask import Flask, request, jsonify
from flask_cors import CORS  # To handle cross-origin requests from iOS
import logging

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

OUTPUT_DIR = "processed_social_media_photos"
MAX_PHOTOS_TO_SELECT = 10 # Maximum photos to process and consider for the album vibe
TARGET_ASPECT_RATIOS = [(4,5)] # Preferred aspect ratios for social media (e.g., Instagram square, portrait)

BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base" # For individual image descriptions (Vision-Encoder-Decoder AI)
GEMMA_LLM_MODEL = "google/gemma-2b-it" # For album caption generation (Large Language Model AI)
AESTHETIC_MODEL_NAME = "HuggingFaceH4/aesthetic-predictor-v2" # For aesthetic scoring (Dedicated AI model)

class AlbumVibeCaptionGenerator:
    def __init__(self):
        logging.info("\nLoading AI models for captioning and vibe generation...")
        # BLIP for individual image descriptions (Vision-Encoder-Decoder)
        self.blip_captioner = pipeline("image-to-text", model=BLIP_MODEL_NAME, device=0 if torch.cuda.is_available() else -1)
        logging.info(f"Loaded BLIP model: {BLIP_MODEL_NAME}")

        # Gemma for LLM-based album caption summarization and vibe generation
        # Ensure you have accepted the model terms on Hugging Face and set up HF_TOKEN secret in Colab
        try:
            self.llm_tokenizer = AutoTokenizer.from_pretrained(GEMMA_LLM_MODEL)
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                GEMMA_LLM_MODEL,
                torch_dtype=torch.float16, # Use float16 for memory efficiency
                device_map="auto", # Automatically map to GPU if available
                token=os.environ.get("HF_TOKEN") # Use HF_TOKEN from Colab Secrets
            )
            logging.info(f"Loaded LLM model: {GEMMA_LLM_MODEL}")
        except Exception as e:
            logging.info(f"ERROR: Could not load LLM model '{GEMMA_LLM_MODEL}'.")
            logging.info("Please ensure you have accepted its terms on Hugging Face and set up your HF_TOKEN secret in Colab.")
            logging.info(f"Details: {e}")
            self.llm_tokenizer = None
            self.llm_model = None

    def generate_individual_captions(self, pil_images):
        """Generates captions for each image using the BLIP model."""
        individual_captions = []
        if self.blip_captioner:
            logging.info("Generating individual image descriptions...")
            for i, img in enumerate(pil_images):
                try:
                    results = self.blip_captioner(img)
                    if results:
                        caption = results[0]['generated_text']
                        individual_captions.append(f"Image {i+1}: {caption}")
                except Exception as e:
                    logging.info(f"  Error captioning image {i+1}: {e}")
                    individual_captions.append(f"Image {i+1}: Could not describe this image.")
        return individual_captions

    def generate_album_vibe_caption(self, individual_captions):
        """
        Generates a single, vibe-fitting caption for the entire album using an LLM.
        """
        if not self.llm_model:
            return "Could not generate album caption due to LLM loading error. Check Colab setup."

        # Craft a prompt for the LLM based on the individual captions
        context = "\n".join(individual_captions)

        prompt = textwrap.dedent(f"""
        You are a highly creative and engaging social media caption generator for influencers.
        You have analyzed a collection of photos from an album.
        Below are individual descriptions of the key photos in the album.

        Based on these descriptions, identify the overall theme, mood, and "vibe" of the entire album.
        Then, generate a single, compelling, and short social media caption (max 3 sentences) that captures this vibe.
        Include relevant emojis and 3-5 popular hashtags to maximize engagement.
        Make it sound authentic and inspiring for social media influencers.

        Individual photo descriptions:
        ---
        {context}
        ---

        Album Vibe Caption:
        """)

        # Tokenize and generate with the LLM

        inputs = self.llm_tokenizer(prompt, return_tensors="pt").to(self.llm_model.device)

        outputs = self.llm_model.generate(
            **inputs,
            max_new_tokens=100, # Adjust for desired length of the album caption
            temperature=0.8,
            do_sample=True,
            pad_token_id=self.llm_tokenizer.eos_token_id
        )

        generated_text = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

        response_start = generated_text.find("Album Vibe Caption:")
        if response_start != -1:
            generated_caption = generated_text[response_start + len("Album Vibe Caption:"):].strip()
        else:
            generated_caption = generated_text

        generated_caption = generated_caption.split("Individual photo descriptions:")[0].strip()

        if not any(tag.startswith('#') for tag in generated_caption.split()):
            generated_caption += "\n#AlbumVibes #InfluencerLife #ContentCreator"

        return generated_caption


class AestheticScorer:
    def __init__(self):
        self.processor = None
        self.model = None
        try:
            # Load feature extractor (for preprocessing images) and the model
            self.processor = AutoFeatureExtractor.from_pretrained(AESTHETIC_MODEL_NAME)
            self.model = AutoModel.from_pretrained(AESTHETIC_MODEL_NAME)
            # Move model to GPU if available
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.model.eval() # Set model to evaluation mode
            logging.info(f"Loaded AI Aesthetic Predictor: {AESTHETIC_MODEL_NAME} on {self.device}.")
        except Exception as e:
            logging.info(f"ERROR: Could not load AI Aesthetic Predictor model '{AESTHETIC_MODEL_NAME}'.")
            logging.info("Please ensure you have all necessary libraries installed (e.g., 'timm').")
            logging.info(f"Details: {e}")
            self.processor = None
            self.model = None
            logging.info("Aesthetic scoring will be skipped or may cause errors.")

    def predict_aesthetic_score(self, pil_image):
        """
        Predicts an aesthetic score for an image using a pre-trained AI model.
        The score typically ranges from 1 to 10.
        """
        if self.processor is None or self.model is None:
            # Fallback to a very basic heuristic or return a default score if model loading failed
            logging.info("Warning: Aesthetic predictor model not loaded. Using fallback score.")
            return 5.0 # Neutral score if AI model failed to load

        try:
            # Preprocess the image using the model's feature extractor
            inputs = self.processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()} # Move inputs to device

            # Perform inference
            with torch.no_grad(): # Disable gradient calculation for inference
                outputs = self.model(**inputs)

            # The model's output is typically a single logit.
            # Convert logit to a score, often scaled to 1-10.
            # For HuggingFaceH4/aesthetic-predictor-v2, a common scaling is to interpret
            # the single logit output as a direct score (e.g., 1-10 or 0-1).
            # If the model outputs logits, a sigmoid or other activation might be needed.
            # Based on common usage, a direct interpretation or clipping is often applied.
            # Let's assume the output is a single float representing the score.
            score = outputs.logits.squeeze().item() # Get the scalar value

            # A common practice is to scale it if the model outputs raw logits or a different range.
            # For this model, scores typically range from 1 to 10. We can clip for safety.
            aesthetic_score = max(1.0, min(10.0, score)) # Ensure score is within 1 to 10 range

            return aesthetic_score

        except Exception as e:
            logging.info(f"Error during aesthetic prediction for image: {e}")
            return 5.0 # Return a neutral score on prediction error


@app.route('/get_image_paths')
def get_image_paths(album_path):
    """Gathers all image and potential Live Photo paths."""
    image_extensions = ['jpg', 'jpeg', 'png', 'webp']
    all_files = os.listdir(album_path)

    photos = []
    live_photos_map = {}

    for filename in all_files:
        name, ext = os.path.splitext(filename)
        ext = ext.lower().lstrip('.')
        full_path = os.path.join(album_path, filename)

        if ext in image_extensions:
            # Check if there's a corresponding .mov for Live Photos
            if os.path.exists(os.path.join(album_path, name + '.mov')):
                if name not in live_photos_map:
                    live_photos_map[name] = {}
                live_photos_map[name]['jpg'] = full_path
            else:
                photos.append({'type': 'still', 'path': full_path})
        elif ext == 'mov':
            if os.path.exists(os.path.join(album_path, name + '.jpg')):
                if name not in live_photos_map:
                    live_photos_map[name] = {}
                live_photos_map[name]['mov'] = full_path

    for name, files in live_photos_map.items():
        if 'jpg' in files and 'mov' in files:
            photos.append({'type': 'live', 'jpg_path': files['jpg'], 'mov_path': files['mov']})
    return photos

def get_image_paths(album_path):
    """Gathers all image and potential Live Photo paths."""
    image_extensions = ['jpg', 'jpeg', 'png', 'webp']
    all_files = os.listdir(album_path)

    photos = []
    live_photos_map = {}

    for filename in all_files:
        name, ext = os.path.splitext(filename)
        ext = ext.lower().lstrip('.')
        full_path = os.path.join(album_path, filename)

        if ext in image_extensions:
            # Check if there's a corresponding .mov for Live Photos
            if os.path.exists(os.path.join(album_path, name + '.mov')):
                if name not in live_photos_map:
                    live_photos_map[name] = {}
                live_photos_map[name]['jpg'] = full_path
            else:
                photos.append({'type': 'still', 'path': full_path})
        elif ext == 'mov':
            if os.path.exists(os.path.join(album_path, name + '.jpg')):
                if name not in live_photos_map:
                    live_photos_map[name] = {}
                live_photos_map[name]['mov'] = full_path

    for name, files in live_photos_map.items():
        if 'jpg' in files and 'mov' in files:
            photos.append({'type': 'live', 'jpg_path': files['jpg'], 'mov_path': files['mov']})

    return photos

def detect_blur(image_np):
    """Calculates a blur score for an image using Laplacian variance."""
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def analyze_faces_for_quality(image_np):
    """
    AI-driven Face Analysis:
    Detects faces and evaluates quality factors like open eyes, smiles.
    This is a conceptual placeholder; real implementations use ML models.
    """
    face_locations = face_recognition.face_locations(image_np)
    if not face_locations:
        return 0, 0 # No faces, no face quality score

    total_face_score = 0

    for face_location in face_locations:
        # A more advanced AI would detect specific expressions, eye closure, etc.
        face_score = 100 # Base score for just having a face

        # Dummy conditions for demonstration, replace with actual ML inference for expressions:
        # e.g., using facial landmarks and a classifier for smiles/open eyes

        total_face_score += face_score

    return len(face_locations), total_face_score

def select_best_live_photo_frame(mov_path):
    """
    Extracts frames from a Live Photo video and selects the "best" one.
    "Best" is determined by an AI-informed heuristic score based on blur and face quality.
    """
    print(f"Processing Live Photo: {mov_path}")
    best_frame = None
    best_score = -1

    try:
        clip = VideoFileClip(mov_path)

        # Process a limited number of frames to avoid excessive computation
        frame_interval = max(1, int(clip.duration / 10)) # Get approx 10 frames

        for i, frame_time in enumerate(np.arange(0, clip.duration, frame_interval)):
            if i >= 20: # Limit total frames processed to 20
                break
            frame_np = clip.get_frame(frame_time) # RGB array

            # Convert to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

            blur_score = detect_blur(frame_bgr)
            num_faces, face_quality_score = analyze_faces_for_quality(frame_bgr)

            # AI-informed scoring heuristic for best frame:
            current_frame_score = blur_score * 0.5 + face_quality_score * 1.0

            if current_frame_score > best_score:
                best_score = current_frame_score
                best_frame = frame_np

        if best_frame is not None:
            return Image.fromarray(best_frame)

    except Exception as e:
        print(f"Error processing Live Photo {mov_path}: {e}")
    return None

def score_photo_engagement(pil_image, aesthetic_scorer):
    """
    Scores a photo based on aesthetic and engagement potential.
    Leverages AI for aesthetic prediction.
    """
    img_np = np.array(pil_image.convert('RGB'))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # 1. AI Aesthetic Score (Most important component for "best shot")
    ai_aesthetic_score = aesthetic_scorer.predict_aesthetic_score(pil_image)

    # 2. Face Presence & Quality Score (AI-informed)
    num_faces, face_quality_score = analyze_faces_for_quality(img_bgr)

    # 3. Basic Photography Metrics (supporting AI)
    blur_score = detect_blur(img_bgr)
    gray_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    brightness_mean = np.mean(gray_image)
    exposure_deviation = abs(brightness_mean - 128)

    # Combine scores with weights. Tune these weights!
    # The AI aesthetic score is now a direct, powerful component.
    total_score = (ai_aesthetic_score * 100) + \
                  (face_quality_score * 1.5) + \
                  (blur_score * 0.1) + \
                  (255 - exposure_deviation)

    return total_score

def apply_social_media_edits(pil_image, preferred_aspect_ratio=(4,5)):
    """Applies common social media edits: crop, enhance, sharpen."""
    img_width, img_height = pil_image.size

    # Smart Cropping (AI for object-aware cropping is a future enhancement)
    target_w, target_h = preferred_aspect_ratio

    potential_h = int(img_width * (target_h / target_w))
    if potential_h <= img_height:
        crop_width = img_width
        crop_height = potential_h
        left = 0
        top = (img_height - crop_height) // 2
    else:
        crop_height = img_height
        crop_width = int(img_height * (target_w / target_h))
        left = (img_width - crop_width) // 2
        top = 0

    right = left + crop_width
    bottom = top + crop_height

    pil_image = pil_image.crop((left, top, right, bottom))

    # Basic Enhancements (can be AI-stylized in future)
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(1.1)

    enhancer = ImageEnhance.Color(pil_image)
    pil_image = enhancer.enhance(1.1)

    pil_image = pil_image.filter(ImageFilter.SHARPEN)

    return pil_image

@app.route('/api/process_album', methods=['POST'])
def process_album_route():
    data = request.get_json()
    album_path = data.get('album_path') if data else None

    if not album_path or not os.path.exists(album_path):
        return jsonify({'error': f"Album path '{album_path}' does not exist or is missing."}), 400

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize AI models
    album_caption_generator = AlbumVibeCaptionGenerator()
    aesthetic_scorer = AestheticScorer()

    all_photos_info = []

    # Gather photos
    photo_paths = get_image_paths(album_path)
    if not photo_paths:
        return jsonify({'error': f"No image files found in '{album_path}'."}), 400

    for i, photo_entry in enumerate(photo_paths):
        original_path = ""
        pil_image = None

        if photo_entry['type'] == 'live':
            original_path = photo_entry['jpg_path']
            best_frame = select_best_live_photo_frame(photo_entry['mov_path'])
            if best_frame:
                pil_image = best_frame
            else:
                continue
        else:  # still photo
            original_path = photo_entry['path']
            try:
                pil_image = Image.open(original_path).convert('RGB')
            except Exception:
                continue

        if pil_image:
            score = score_photo_engagement(pil_image, aesthetic_scorer)
            all_photos_info.append({
                'original_path': original_path,
                'score': score,
                'pil_image': pil_image  # Will be removed before returning JSON
            })

    all_photos_info.sort(key=lambda x: x['score'], reverse=True)
    selected_photos_info = all_photos_info[:MAX_PHOTOS_TO_SELECT]

    if not selected_photos_info:
        return jsonify({'error': "No suitable photos found or selected."}), 400

    processed_pil_images_for_captioning = []

    saved_files = []

    for idx, photo_data in enumerate(selected_photos_info):
        original_img_path = photo_data['original_path']
        edited_pil_image = apply_social_media_edits(photo_data['pil_image'], preferred_aspect_ratio=TARGET_ASPECT_RATIOS[0])
        output_filename = f"post_{idx+1}_{os.path.basename(original_img_path)}"
        output_filepath = os.path.join(OUTPUT_DIR, output_filename)
        edited_pil_image.save(output_filepath)
        processed_pil_images_for_captioning.append(edited_pil_image)
        saved_files.append(output_filepath)

    individual_blip_captions = album_caption_generator.generate_individual_captions(processed_pil_images_for_captioning)
    album_vibe_caption = album_caption_generator.generate_album_vibe_caption(individual_blip_captions)

    # Optionally write caption to file (or skip in API)
    album_caption_filepath = os.path.join(OUTPUT_DIR, "album_vibe_caption.txt")
    with open(album_caption_filepath, 'w', encoding='utf-8') as f_album_caption:
        f_album_caption.write("--- Album Vibe Caption for All Selected Photos ---\n\n")
        f_album_caption.write(textwrap.fill(album_vibe_caption, width=80))
        f_album_caption.write("\n\n--- Individual Photo Descriptions (for reference) ---\n\n")
        for cap in individual_blip_captions:
            f_album_caption.write(textwrap.fill(cap, width=80) + "\n")

    # Remove PIL images before returning JSON (not serializable)
    for p in selected_photos_info:
        p.pop('pil_image', None)

    return jsonify({
        'saved_files': saved_files,
        'album_vibe_caption': album_vibe_caption,
        'individual_captions': individual_blip_captions,
        'caption_file': album_caption_filepath
    })