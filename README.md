# airdrawback

📦 AirDraw Backend – FastAPI + AI Sketch Classifier
This is the backend component of the AirDraw project – an interactive AI-powered sketch recognition system where users can draw in the air using their fingers, and get real-time predictions powered by a CNN model trained on the QuickDraw dataset.

🚀 Features
Accepts base64 or file-uploaded sketch images

Processes and resizes sketches to 28x28 grayscale

Loads a pretrained CNN model (trained on 20 QuickDraw classes)

Returns prediction result with class label and corresponding real image

CORS enabled to support frontend integration

🧠 Model Info
Dataset: QuickDraw (20 selected categories)

Samples: 10,000 sketches per category → 200,000 total

Architecture: Lightweight CNN (4 conv layers, batch norm, dropout)

Accuracy: ~90% validation accuracy

Format: model.h5
