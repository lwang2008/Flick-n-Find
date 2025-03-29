import streamlit as st
import os
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from ultralytics import YOLO
from datetime import datetime
from collections import Counter
import tensorflow as tf
st.title("Upload")

yolo_model = YOLO('yolov8n.pt')  

imageDB = pd.DataFrame(columns = ["ind", "filepath", "keywords", "location", "date", "time", "contact", "status"])

def basicKeyword(image):
    
    # color, brand(optional), type
    cv_image = np.array(image)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

    results = yolo_model(cv_image)

    keywords = set()
    
    for box in results[0].boxes:
        if box.conf > 0.5:  
            class_id = int(box.cls)
            class_name = yolo_model.names[class_id]
            keywords.add(class_name)  # Add to set to ensure uniqueness

    keyword = " ".join(keywords)
    
    if keyword == "person":
        keyword = "sweatshirt"
    return keyword

# Expanded color ranges
COLOR_RANGES = {
    "Red": [(0, 50, 50), (10, 255, 255)],
    "Dark Red": [(170, 50, 50), (180, 255, 255)],
    "Green": [(35, 50, 50), (85, 255, 255)],
    "Light Green": [(85, 50, 50), (100, 255, 255)],
    "Blue": [(100, 50, 50), (140, 255, 255)],
    "Sky Blue": [(90, 50, 50), (100, 255, 255)],
    "Yellow": [(20, 100, 100), (30, 255, 255)],
    "Orange": [(10, 100, 100), (20, 255, 255)],
    "Brown": [(10, 100, 50), (20, 150, 150)],
    "Purple": [(140, 100, 100), (160, 255, 255)],
    "Pink": [(160, 100, 100), (170, 255, 255)],
    "White": [(0, 0, 200), (180, 30, 255)],
    "Black": [(0, 0, 0), (180, 255, 50)],
    "Gray": [(0, 0, 50), (180, 50, 200)],
    "Teal": [(85, 50, 50), (95, 255, 200)],
    "Magenta": [(140, 50, 50), (150, 255, 255)],
    "Beige": [(20, 50, 150), (30, 100, 200)],
}

def colorKeyword(image):
    # Load YOLO model
    model = YOLO('yolov8n.pt')  

    # Perform object detection
    results = model.predict(image, save=False, save_txt=False, conf=0.5)

    if results and len(results[0].boxes):
        box = results[0].boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        roi = image[y1:y2, x1:x2]

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        color_counts = Counter()
        for color_name, (lower, upper) in COLOR_RANGES.items():
            mask = cv2.inRange(hsv_roi, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
            count = cv2.countNonZero(mask)
            if count > 0:
                color_counts[color_name] += count

        if color_counts:
            return color_counts.most_common(1)[0][0]

    return ""

MODEL_PATH = 'frozen_inference_graph.pb'  


LABEL_MAP = {
    1: "Adidas",
    2: "Apple",
    3: "BMW",
    4: "Citroen",
    5: "Cocacola",
    6: "DHL",
    7: "Fedex",
    8: 'Ferrari',
    9: 'Ford',
    10: 'Google',
    11: 'HP',
    12: 'Heineken',
    13: "Intel",
    14: "McDonalds",
    15: "Mini",
    16: "Nbc",
    17: "Nike",
    18: "Pepsi",
    19: "Porsche",
    20: "Puma",
    21: "RedBull",
    22: "Sprite",
    23: "Starbucks",
    24: "Texaco",
    25: "Unicef",
    26: "Vodafone",
    27: "Yahoo" 
}

def load_frozen_graph(model_path):
    
    graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(model_path, 'rb') as fid:
        serialized_graph = fid.read()
        graph_def.ParseFromString(serialized_graph)
    
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, name='')
    
    return graph

def logoKeyword(model_path, uploaded_image):

    try:
        detection_graph = load_frozen_graph(model_path)
        
        image = Image.open(uploaded_image)
        image_np = np.array(image)
        
        image_np_expanded = np.expand_dims(image_np, axis=0)
        
        with detection_graph.as_default():
            with tf.compat.v1.Session(graph=detection_graph) as sess:
                ops = detection_graph.get_operations()
                
                # Find tensor names
                tensor_names = [op.name for op in ops]
                
                # find correct tensor names
                image_tensor_name = [name for name in tensor_names if 'image_tensor' in name.lower()]
                boxes_tensor_name = [name for name in tensor_names if 'detection_boxes' in name.lower()]
                scores_tensor_name = [name for name in tensor_names if 'detection_scores' in name.lower()]
                classes_tensor_name = [name for name in tensor_names if 'detection_classes' in name.lower()]
                num_detections_tensor_name = [name for name in tensor_names if 'num_detections' in name.lower()]
                
                # Get tensors
                image_tensor = detection_graph.get_tensor_by_name(f'{image_tensor_name[0]}:0')
                boxes = detection_graph.get_tensor_by_name(f'{boxes_tensor_name[0]}:0')
                scores = detection_graph.get_tensor_by_name(f'{scores_tensor_name[0]}:0')
                classes = detection_graph.get_tensor_by_name(f'{classes_tensor_name[0]}:0')
                num_detections = detection_graph.get_tensor_by_name(f'{num_detections_tensor_name[0]}:0')
                
                # Run detection
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded}
                )
                
                # Process results
                logo_details = ""
                for i in range(int(num_detections[0])):
                    if scores[0][i] > 0.5: 
                        class_id = int(classes[0][i])
                        logo_name = LABEL_MAP.get(class_id, f'Unknown Logo {class_id}')
                        logo_details += logo_name
                        
                
                return logo_details
    
    except Exception as e:
        #st.error(f"Error in logo detection: {e}")
        return ""


# adding entry to dataframe
def addEntry(name, image, keywords, contact):
    file_path = "images/" + name
    current_datetime = datetime.now()
    imageDB = pd.read_csv("imageDB.csv")
    new_entry = pd.DataFrame({"ind": len(imageDB),
                "filepath": file_path,
                "keywords": [keywords],
                 "location": "Palo Alto, CA",
                 "date": current_datetime.date(),
                 "time": current_datetime.strftime("%I:%M:%S %p"),
                 "contact": contact,
                 "status": "lost"})
    imageDB = pd.concat([imageDB, new_entry], ignore_index=True)
    imageDB.to_csv("imageDB.csv", index=False)


def entries():
    return len(imageDB)


st.header("Upload Lost Item")
uploaded_image = st.file_uploader("Upload an image of the lost item:", type=["jpg", "png", "jpeg"])
contact = st.text_input("Enter your number:")

if st.button("Submit"):
    if uploaded_image:
        imageDB = pd.read_csv("imageDB.csv")
        name = "img" + str(len(imageDB)) + ".png"
        file_path = os.path.join("images", name)
        with open(file_path, "wb") as f:
            f.write(uploaded_image.getbuffer())
        
        uploaded_image.seek(0)  
        image_pil = Image.open(uploaded_image)
        image_cv = np.array(image_pil)  
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)  

        
        keyword_basic = basicKeyword(image_pil)
        keyword_color = colorKeyword(image_cv)
        keyword_logos = logoKeyword(MODEL_PATH, uploaded_image)

        keywords = keyword_color + " " + keyword_logos + " " + keyword_basic

        if contact:
            addEntry(name, image_pil, keywords, str(int(contact)))
        else:
            addEntry(name, image_pil, keywords, None)
        
        st.success(f"Item added successfully with keywords: " + keywords)
        
        imageDB = pd.read_csv("imageDB.csv")
        st.write(imageDB)
        csv_file_path = "imageDB.csv"
        imageDB.to_csv(csv_file_path, index=False)
    else:
        st.error("Please provide both an image and a contact.")