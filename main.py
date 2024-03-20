import pyautogui
import os
import subprocess
import time
import shutil
import cv2
import numpy as np
from skimage import metrics
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel
import cv2
from PIL import Image, ImageDraw, ImageFont
import warnings


warnings.filterwarnings('ignore') 


def capture_screenshot():
    pyautogui.screenshot(os.path.join(os.getcwd(), "input_images", "input_image.png"))
    return True


def identify_all_icons():
    shutil.rmtree(os.path.join(os.getcwd(), "output_images"))
    ## update with the mentioned paths
    cmd = f"""cd venv/Scripts && activate && python yolo\\content\\yolov5\\detect.py --weights yolov5\\content\\yolov5\\runs\\train\\exp4\\weights\\best.pt --img 640  --conf 0.4 --project output_images --save-txt --save-conf --source input_images\\input_image.png"""
    subprocess.call(cmd, shell=True)    


def parse_annotations(annotation_file, img_width, img_height):
    with open(annotation_file, 'r') as f:
        lines = f.readlines()
    annotations = []
    for line in lines:
        parts = line.strip().split(' ')
        label = int(parts[0])
        x_center, y_center, width, height = map(float, parts[1:-1])
        x1, y1 = int((x_center - width/2) * img_width), int((y_center - height/2) * img_height)
        x2, y2 = int((x_center + width/2) * img_width), int((y_center + height/2) * img_height)
        annotations.append((label, x1, y1, x2, y2))
    return annotations


def draw_boxes(image_path, boxes, output_path):
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)
    image_width, image_height = pil_image.size

    # Draw bounding boxes and write class names
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.truetype("arial.ttf", size=20) 

    for box in boxes:
        x_center, y_center, width, height, class_name = box
        if class_name is not None:
            x_min = x_center
            y_min = y_center
            x_max = width 
            y_max = height 
            print(x_center)
            print(x_min, y_min, x_max, y_max)

            # Draw bounding box
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)

            # Write class name
            draw.text(((x_min + x_max) // 2, y_min - 20), class_name.capitalize(), fill="red", font=font)
            # draw.text((x_min, y_min), class_name, fill="red", font=font)

    pil_image.save(output_path)


def retrieve_embeddings(image_path, input_text_list, use_inputs):
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    image = Image.open(image_path)
    inputs = processor(text=input_text_list, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    embeds1 = outputs.image_embeds
    if use_inputs:
        logits_per_image = outputs.logits_per_image # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
        print(probs)
    return embeds1


def calculate_image_similarity(image1_path, image2_path, input_text_list):
    image1_embeddings = retrieve_embeddings(image1_path, input_text_list, use_inputs=False)
    image2_embeddings = retrieve_embeddings(image2_path, input_text_list, use_inputs=False)
    a = image1_embeddings[0].detach().numpy()
    b = image2_embeddings[0].detach().numpy()
    a = a.reshape(1, -1)
    b = b.reshape(1, -1)
    cosine_sim = cosine_similarity(a, b)
    return cosine_sim[0][0]


def check_similarity():
    # Load the image
    snapshot_image_path = os.path.join(os.getcwd(), "input_images/input_image.png")
    snapshot_image = cv2.imread(snapshot_image_path)
    img_height, img_width, _ = snapshot_image.shape
    annotation_file_path = os.path.join(os.getcwd(), "output_images/exp/labels/input_image.txt")
    annotations = parse_annotations(annotation_file_path, img_width, img_height)
    reference_icons_path = os.path.join(os.getcwd(), "reference_icons")

    final_boxes = []
    count = 1
    for label, x1, y1, x2, y2 in annotations:
        print(f"Icon analysis: {count}/{len(annotations)} in progress")
        count +=1
        cropped_icon = snapshot_image[y1:y2, x1:x2]
        stretch_factor = 6
        new_h = int(cropped_icon.shape[0] * stretch_factor)
        new_w = int(cropped_icon.shape[1] * stretch_factor)
        cropped_icon = cv2.resize(cropped_icon, (new_w, new_h))

        scores_list = []    
        for icon in [i for i in os.listdir(reference_icons_path) if i.endswith(".png")]:
            reference_icon = cv2.imread(os.path.join(reference_icons_path, icon))
            # cropped_icon = cv2.resize(cropped_icon, (reference_icon.shape[1], reference_icon.shape[0]), interpolation = cv2.INTER_AREA)
            cv2.imwrite(os.path.join(os.getcwd(), "temp_folder/reference_icon.png"), reference_icon)
            cv2.imwrite(os.path.join(os.getcwd(), "temp_folder/cropped_icon.png"), cropped_icon)
            
            shape_similarity = calculate_image_similarity(
                        os.path.join(os.getcwd(), "temp_folder/cropped_icon.png"), 
                        os.path.join(os.getcwd(), "temp_folder/reference_icon.png"),
                        ["Hospital icon", "restaurant icon", "bank icon", "icon with green dot", "museum icon", "supermarket"]
                    )
            scores_list.append((icon, shape_similarity))
        scores_list = sorted(
            scores_list,
            key=lambda x: x[1],
            reverse=True
        )
        # cv2.imshow('cropped_image', cropped_icon)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        print(scores_list)
        print(scores_list[0])
        class_identified = None
        if scores_list[0][1]>0.90:
            class_identified = scores_list[0][0].split("_")[0]

        final_boxes.append((x1, y1, x2, y2, class_identified))

    draw_boxes(r"input_images\input_image.png", final_boxes,"output_image.png",)

if __name__=="__main__":
    time.sleep(1)
    capture_screenshot()
    identify_all_icons()
    check_similarity()