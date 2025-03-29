# FlicknFind

Flick n Find is a community-wide lost-and-found platform powered by AI, making it easy to report, search, and reconnect lost items with their owners. Say goodbye to cluttered feeds and hello to effortless recovery! On community apps like Nextdoor, there is a slight chance you find your lost item on your feed. We wanted to make it easy to search for a lost item.

This placed 2nd place at GunnHacks 11.0. Checkout our [devpost here](tinyurl.com/FlicknFind).

## Team Members

- Lucas Wang
- Aakash Koneru
- Neeraj Gummalam

## Features

- AI-powered image analysis using YOLO and Universal Sentence Encoder
- Natural language search for lost items
- Easy upload of found items with automatic keyword generation
- Community-based lost and found platform
- Real-time status updates for items

## Setup

1. Clone the repository:
```bash
git clone https://github.com/lwang2008/Flick-n-Find.git
cd Flick-n-Find
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Download the required model files:
   - YOLO model (yolov8n.pt):
     ```bash
     wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
     ```
   - Universal Sentence Encoder model will be downloaded automatically when you first run the application

5. Run the application:
```bash
streamlit run home.py
```

The application will be available at http://localhost:8501

## Usage

1. **Uploading Found Items**:
   - Click "I Found Something"
   - Upload an image of the found item
   - The system will automatically analyze the image and generate keywords
   - Add your contact information
   - Submit the item

2. **Searching for Lost Items**:
   - Click "I Lost Something"
   - Enter a description of your lost item
   - The system will match your description with items in the database
   - Contact the finder if you see your item

## Technologies Used

- Streamlit for the web interface
- YOLO for object detection
- Universal Sentence Encoder for semantic search
- OpenCV for image processing
- TensorFlow for AI models
- Pandas for data management

## Project Structure

```
FlicknFind/
├── home.py              # Main application file
├── pages/              # Streamlit pages
│   ├── upload.py       # Upload page
│   └── search.py       # Search page
├── preprocess.py       # Text preprocessing
├── keywordmatching.py  # Keyword matching logic
├── images/            # Directory for uploaded images
├── imageDB.csv        # Database of items
├── yolov8n.pt         # YOLO model file (download required)
└── requirements.txt   # Python dependencies
```

## Model Files

The project requires two main model files:

1. **YOLO Model (yolov8n.pt)**
   - Used for object detection in images
   - Download from: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
   - Place in the root directory of the project

2. **Universal Sentence Encoder**
   - Used for semantic search functionality
   - Automatically downloaded when first running the application
   - No manual download required

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

Special thanks to all team members who contributed to this project:
- Lucas Wang
- Aakash Koneru
- Neeraj Gummalam 

