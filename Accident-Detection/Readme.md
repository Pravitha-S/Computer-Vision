# Accident Detection (YOLOv8)

This project detects potential collisions in traffic video with:
- YOLOv8 for object detection & tracking
- Heuristics for overlap/proximity → collision detection
- Gradio web UI for upload + chat with agent

## Quick start
1. Install requirements: `pip install -r requirements.txt`
2. Place video to test in `sample_videos/` or upload via UI.
3. Run: `python run_analysis.py` 

## Demo Screenshot
![Accident Detection Demo](assets/accidents_demo.png)
