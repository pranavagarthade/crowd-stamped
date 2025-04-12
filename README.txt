# 🧠 Crowd Management & Stampede Prevention System

A real-time AI-based system designed to monitor crowd density, detect potential stampede risks, and suggest safer redirection paths—built as part of the **Ashwamedh Hackathon 2025**, where it placed unofficially in the **Top 3**!

## 🏁 Overview

Large crowds at public events can pose serious safety challenges. Our solution leverages computer vision and real-time analytics to:
- Detect and count individuals in live video feeds
- Analyze crowd density at multiple exits
- Identify overcrowded zones
- Suggest redirection paths to prevent stampedes
- Record peak crowd counts and timestamps for future analysis

## 🔧 Tech Stack

- **Python**
- **Flask** – Backend framework
- **OpenCV** – For video feed handling
- **YOLOv8** – Real-time object detection
- **SocketIO** – For real-time updates between backend and frontend
- **Matplotlib** – Visualization of crowd trends
- **HTML/CSS/JS** – Frontend interface

## 📁 Features

✅ Live camera feeds from 3 simulated exits  
✅ Real-time people counting with YOLOv8  
✅ Dynamic crowd risk levels (Low, Medium, High)  
✅ Redirection suggestions when exits are overcrowded  
✅ Historical data logging with peak timestamps  
✅ One-click trend graph generation

## 📈 Future Scope

- The system is designed to generate and store a `crowd_data.csv` file containing **crowd counts and timestamps** at regular intervals.
- This file will serve as a **future expansion** to build trend-based analytics and visualizations showing how crowd density fluctuates over time.
- This can help in:
  - Predicting peak hours
  - Planning more efficient crowd control strategies
  - Enhancing real-time alert systems

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/crowd-management-system.git
   cd crowd-management-system
