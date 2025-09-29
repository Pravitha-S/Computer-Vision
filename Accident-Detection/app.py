# ==================== INSTALL DEPENDENCIES ====================
# !pip install transformers torch torchvision opencv-python pillow gradio -q
# !pip install ultralytics supervision -q
# !pip install langchain -q

import cv2
import torch
import numpy as np
import gradio as gr
import json
from datetime import datetime
import supervision as sv
from ultralytics import YOLO
import time

# ==================== ACCURATE COLLISION DETECTOR ====================
class AccurateCollisionDetector:
    def __init__(self):
        self.model = YOLO('yolov8m.pt')
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        self.collision_threshold = 0.7
        
    def detect_collision_frames(self, video_path):
        """Accurate collision detection with proper timing"""
        if video_path is None:
            return []
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        
        # Get actual video FPS and duration
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30  # Default fallback
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps if fps > 0 else 0
        
        collision_events = []
        frame_count = 0
        collision_id = 0
        active_collisions = {}
        
        frame_skip = 2  # Process every 2nd frame for speed
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue
            
            # Calculate actual timestamp based on real FPS
            actual_timestamp = frame_count / fps
            
            # Skip processing if beyond reasonable time
            if actual_timestamp > 10:  # Max 10 seconds for demo
                break
            
            results = self.model.track(frame, persist=True, verbose=False)
            
            if len(results) > 0 and hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy()
                
                current_vehicles = {}
                for i, (box, cls, track_id) in enumerate(zip(boxes, classes, track_ids)):
                    if cls in self.vehicle_classes:
                        current_vehicles[track_id] = {
                            'box': box,
                            'class': cls,
                            'class_name': self.get_class_name(cls)
                        }
                
                # Detect collisions in current frame
                frame_collisions = self.detect_frame_collisions(current_vehicles)
                
                for collision in frame_collisions:
                    collision_key = tuple(sorted(collision['vehicles']))
                    
                    if collision_key not in active_collisions:
                        collision_id += 1
                        active_collisions[collision_key] = collision_id
                        
                        collision_events.append({
                            'collision_id': collision_id,
                            'frame': frame_count,
                            'timestamp': actual_timestamp,  # Real timestamp based on FPS
                            'confidence': collision['confidence'],
                            'vehicles_involved': len(collision['vehicles']),
                            'vehicle_types': collision['vehicle_types'],
                            'collision_type': collision['type']
                        })
                
                # Clean old collisions
                active_collisions = {k: v for k, v in active_collisions.items() 
                                   if any(ev['collision_id'] == v and 
                                         actual_timestamp - ev['timestamp'] < 2.0 
                                         for ev in collision_events)}
            
            frame_count += 1
        
        cap.release()
        
        # Filter collisions to only include those within video duration
        valid_collisions = [c for c in collision_events if c['timestamp'] <= video_duration]
        
        return valid_collisions
    
    def detect_frame_collisions(self, vehicles):
        """Detect collisions with accurate overlap checking"""
        collisions = []
        vehicle_list = list(vehicles.items())
        
        for i in range(len(vehicle_list)):
            for j in range(i + 1, len(vehicle_list)):
                track_id1, veh1 = vehicle_list[i]
                track_id2, veh2 = vehicle_list[j]
                
                overlap_ratio = self.calculate_overlap_ratio(veh1['box'], veh2['box'])
                
                if overlap_ratio > 0.35:  # Realistic collision threshold
                    confidence = min(0.7 + (overlap_ratio * 0.5), 0.95)
                    
                    collisions.append({
                        'vehicles': [track_id1, track_id2],
                        'vehicle_types': [veh1['class_name'], veh2['class_name']],
                        'type': "collision",
                        'confidence': confidence,
                        'overlap': overlap_ratio
                    })
        
        return collisions
    
    def calculate_overlap_ratio(self, box1, box2):
        """Calculate accurate overlap ratio"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x1 < x2 and y1 < y2:
            intersection = (x2 - x1) * (y2 - y1)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            smaller_area = min(area1, area2)
            return intersection / smaller_area if smaller_area > 0 else 0
        return 0.0
    
    def get_class_name(self, class_id):
        class_names = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
        return class_names.get(class_id, "vehicle")

class AgenticAISystem:
    def __init__(self):
        self.analysis_history = []
        self.collision_data = None
        
    def set_collision_data(self, data):
        self.collision_data = data
        self.analysis_history.append({
            'timestamp': datetime.now(),
            'collisions_detected': len(data) if data else 0,
            'data': data
        })
    
    def analyze_collision_patterns(self):
        """Agentic analysis of collision patterns"""
        if not self.collision_data:
            return "No collision data available for analysis."
        
        collisions = self.collision_data
        total_collisions = len(collisions)
        
        if total_collisions == 0:
            return "Agent Analysis: No collision patterns detected. Traffic flow appears normal."
        
        # Calculate patterns
        timestamps = [c['timestamp'] for c in collisions]
        time_intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        
        analysis = "🤖 **AGENTIC AI ANALYSIS**\n\n"
        analysis += f"📊 **Pattern Detection**\n"
        analysis += f"• Total Collisions: {total_collisions}\n"
        
        if len(timestamps) > 0:
            analysis += f"• Time Range: {min(timestamps):.2f}s - {max(timestamps):.2f}s\n"
        
        if time_intervals:
            avg_interval = sum(time_intervals) / len(time_intervals)
            analysis += f"• Average Time Between Events: {avg_interval:.2f}s\n"
        
        # Risk assessment
        if total_collisions >= 5:
            analysis += "• 🚨 HIGH RISK: Multiple consecutive collisions detected\n"
        elif total_collisions >= 3:
            analysis += "• ⚠️ MEDIUM RISK: Several collision events\n"
        else:
            analysis += "• ✅ LOW RISK: Isolated incident(s)\n"
        
        # Temporal analysis
        if len(timestamps) > 1:
            if max(time_intervals) < 1.0:
                analysis += "• ⏰ RAPID SUCCESSION: Events occurring quickly\n"
        
        analysis += "\n🔍 **Agent Recommendation**\n"
        if total_collisions >= 3:
            analysis += "Immediate emergency response recommended. Multiple vehicles involved in chain reaction."
        else:
            analysis += "Standard incident response. Isolated collision event."
        
        return analysis
    
    def answer_question(self, question):
        """Intelligent question answering with agentic reasoning"""
        if not self.collision_data:
            return "Please analyze a video first to get collision data."
        
        question_lower = question.lower()
        collisions = self.collision_data
        
        # Get accurate collision information
        total_collisions = len(collisions)
        timestamps = [c['timestamp'] for c in collisions]
        
        if "second collision" in question_lower or "2nd collision" in question_lower:
            if total_collisions >= 2:
                return f"🕒 The second collision occurred at {timestamps[1]:.2f} seconds."
            else:
                return f"Only {total_collisions} collision(s) detected. No second collision available."
        
        elif "first collision" in question_lower:
            if total_collisions >= 1:
                return f"🕒 The first collision occurred at {timestamps[0]:.2f} seconds."
            else:
                return "No collisions detected."
        
        elif "how many collision" in question_lower:
            return f"📊 I detected {total_collisions} collision events in the video."
        
        elif "last collision" in question_lower:
            if total_collisions >= 1:
                return f"🕒 The last collision occurred at {timestamps[-1]:.2f} seconds."
            else:
                return "No collisions detected."
        
        elif "time between" in question_lower or "interval" in question_lower:
            if total_collisions >= 2:
                intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
                avg_interval = sum(intervals) / len(intervals)
                return f"⏱️ Average time between collisions: {avg_interval:.2f} seconds"
            else:
                return "Need at least 2 collisions to calculate time intervals."
        
        elif "pattern" in question_lower or "analysis" in question_lower:
            return self.analyze_collision_patterns()
        
        elif "vehicles" in question_lower:
            if total_collisions >= 1:
                vehicles = collisions[0]['vehicles_involved']
                types = collisions[0].get('vehicle_types', ['vehicles'])
                return f"🚗 First collision involved {vehicles} vehicles: {', '.join(types)}"
            return "No vehicle data available."
        
        else:
            return self.analyze_collision_patterns()

# ==================== SIMPLE REPORT GENERATOR ====================
class SimpleReportGenerator:
    def generate_report(self, collision_data, video_duration=0):
        if not collision_data:
            return self.create_no_collision_report(video_duration)
        
        total_collisions = len(collision_data)
        first_collision = collision_data[0]
        max_confidence = max([c['confidence'] for c in collision_data])
        
        report = f"""
🎯 **COLLISION ANALYSIS REPORT**

📊 **SUMMARY**
• Total Collisions: {total_collisions}
• First Collision: {first_collision['timestamp']:.2f}s
• Video Duration: {video_duration:.2f}s
• Highest Confidence: {max_confidence:.2f}/1.00

🕒 **COLLISION EVENTS**
"""
        
        # Show collisions with real timestamps
        for i, collision in enumerate(collision_data[:6]):  # Show max 6
            report += f"• {collision['timestamp']:.2f}s - {collision['vehicles_involved']} vehicles\n"
        
        if total_collisions > 6:
            report += f"• ... and {total_collisions - 6} more events\n"
        
        report += f"""
⚠️ **RECOMMENDATIONS**
• Review footage carefully
• Verify {total_collisions} detected events
• Check timestamps against actual video
"""
        return report
    
    def create_no_collision_report(self, video_duration):
        return f"""
✅ **SAFETY ANALYSIS REPORT**

📊 **RESULTS**
• Collisions Detected: 0
• Video Duration: {video_duration:.2f}s
• Status: No incidents found

🎯 **CONCLUSION**
No collision activity detected.
"""

# ==================== MAIN SYSTEM ====================
collision_detector = AccurateCollisionDetector()
report_generator = SimpleReportGenerator()
ai_agent = AgenticAISystem()

def analyze_video_only(video_path):
    if video_path is None:
        return "Please upload a video file first!"
        
    start_time = time.time()
    
    # Process video
    collision_data = collision_detector.detect_collision_frames(video_path)
    
    # Get video duration
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps if fps > 0 else 0
    cap.release()
    
    # Generate report
    report = report_generator.generate_report(collision_data, video_duration)
    
    # Store data for AI agent
    ai_agent.set_collision_data(collision_data)
    
    processing_time = time.time() - start_time
    
    output = f"""🎯 **ANALYSIS COMPLETE**
⏱️ Processing Time: {processing_time:.1f}s
📹 Video Duration: {video_duration:.2f}s

{report}

💬 **AI Agent Ready** - Ask questions below!"""
    return output

def chat_about_video(question):
    return ai_agent.answer_question(question)

# ==================== GRADIO UI ====================
def create_interface():
    css = """
    .gradio-container {
        background: #000000 !important;
    }
    .dark {
        background: #000000 !important;
    }
    .panel {
        background: #1a1a1a !important;
        border: 1px solid #333 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .button-primary {
        background: linear-gradient(45deg, #8B5CF6, #7C3AED) !important;
        border: none !important;
        color: white !important;
    }
    .text-white {
        color: white !important;
    }
    """

    with gr.Blocks(title="AI Car Collision Detection", css=css, theme=gr.themes.Default(primary_hue="purple")) as demo:
        
        gr.Markdown("<div style='text-align: center;'><h1 style='color: #8B5CF6;'>🚗 AI Car Collision Detection</h1></div>")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Video Analysis Section
                gr.Markdown("<div class='panel'><h2 style='color: #8B5CF6;'>📹 Video Analysis</h2><p class='text-white'>Upload traffic video</p>")
                video_input = gr.Video(label="Upload Video")
                analyze_btn = gr.Button("🔍 Analyze & Report", variant="primary", elem_classes="button-primary")
                gr.Markdown("</div>")
                
                # Analysis Report
                gr.Markdown("<div class='panel'><h2 style='color: #8B5CF6;'>📊 Analysis Report</h2>")
                output_text = gr.Textbox(label="Report", lines=15, show_copy_button=True)
                gr.Markdown("</div>")
            
            with gr.Column(scale=1):
                # Chat Section
                gr.Markdown("<div class='panel'><h2 style='color: #8B5CF6;'>💬 Ask Questions</h2><p class='text-white'>Question about the video:</p>")
                question_input = gr.Textbox(label="Your Question", placeholder="Ask about collisions, timing, patterns...", lines=2)
                chat_btn = gr.Button("🤖 Ask AI Agent", variant="primary", elem_classes="button-primary")
                gr.Markdown("</div>")
                
                # Quick Questions
                gr.Markdown("<div class='panel'><h3 style='color: #8B5CF6;'>💡 Quick Questions</h3><p class='text-white'>Click to ask:</p>")
                with gr.Row():
                    btn_first = gr.Button("🕒 First Collision", elem_classes="button-primary")
                    btn_second = gr.Button("2️⃣ Second Collision", elem_classes="button-primary")
                    btn_total = gr.Button("📊 Total Collisions", elem_classes="button-primary")
                gr.Markdown("</div>")
                
                # Chat Response
                gr.Markdown("<div class='panel'><h2 style='color: #8B5CF6;'>💭 AI Response</h2>")
                chat_output = gr.Textbox(label="Agent Response", lines=8, show_copy_button=True)
                gr.Markdown("</div>")
        
        # Instructions
        gr.Markdown("""
        <div class='panel'>
        <h3 style='color: #8B5CF6;'>🎯 How to Use</h3>
        <p class='text-white'>1. Upload traffic video</p>
        <p class='text-white'>2. Click 'Analyze & Report'</p>
        <p class='text-white'>3. Read analysis report</p>
        <p class='text-white'>4. Ask questions using chat or quick buttons</p>
        </div>
        """)
        
        # Event handlers
        analyze_btn.click(analyze_video_only, inputs=[video_input], outputs=[output_text])
        chat_btn.click(chat_about_video, inputs=[question_input], outputs=[chat_output])
        
        btn_first.click(lambda: chat_about_video("first collision"), outputs=[chat_output])
        btn_second.click(lambda: chat_about_video("second collision"), outputs=[chat_output])
        btn_total.click(lambda: chat_about_video("how many collisions"), outputs=[chat_output])
    
    return demo

# ==================== LAUNCH ====================
print("🚀 Launching AI-Powered Collision Detection...")
print("🤖 Features: YOLOv8 + Agentic AI + Accurate Timing")

demo = create_interface()
demo.launch(share=True, debug=True)