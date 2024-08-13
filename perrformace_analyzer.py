import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import requests
from typing import List, Dict

# Replace with your actual Claude API key
CLAUDE_API_KEY = "your_claude_api_key_here"
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"

def load_csv_data(file_path: str) -> pd.DataFrame:
    """Load CSV file and return a pandas DataFrame."""
    return pd.read_csv(file_path)

def analyze_arm_angles(df: pd.DataFrame) -> Dict[str, float]:
    """Analyze arm angles and return statistics."""
    left_angles = df['Left Arm Angle']
    right_angles = df['Right Arm Angle']
    
    return {
        'left_mean': left_angles.mean(),
        'left_std': left_angles.std(),
        'right_mean': right_angles.mean(),
        'right_std': right_angles.std(),
        'symmetry': np.abs(left_angles.mean() - right_angles.mean())
    }

def create_angle_plots(df: pd.DataFrame) -> List[str]:
    """Create plots for arm angles and save them as  .png formate in plots folder."""
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='Frame', y='Left Arm Angle', label='Left Arm')
    sns.lineplot(data=df, x='Frame', y='Right Arm Angle', label='Right Arm')
    plt.title('Arm Angles Over Time')
    plt.xlabel('Frame')
    plt.ylabel('Angle (degrees)')
    plt.legend()
    
    plot_path = 'arm_angles_plot.png'
    plt.savefig(plot_path)
    plt.close()
    
    return [plot_path]

def analyze_key_frames(frame_folder: str, df: pd.DataFrame) -> List[Dict[str, any]]:
    """Analyze key frames and return insights."""
    key_frames = []
    
    # Find frames with extreme angles
    max_left_frame = df.loc[df['Left Arm Angle'].idxmax()]
    max_right_frame = df.loc[df['Right Arm Angle'].idxmax()]
    min_left_frame = df.loc[df['Left Arm Angle'].idxmin()]
    min_right_frame = df.loc[df['Right Arm Angle'].idxmin()]
    
    for frame_data in [max_left_frame, max_right_frame, min_left_frame, min_right_frame]:
        frame_number = int(frame_data['Frame'])
        frame_path = os.path.join(frame_folder, f"frame_{frame_number}.jpg")
        
        if os.path.exists(frame_path):
            key_frames.append({
                'frame_number': frame_number,
                'left_angle': frame_data['Left Arm Angle'],
                'right_angle': frame_data['Right Arm Angle'],
                'frame_path': frame_path
            })
    
    return key_frames

def get_claude_analysis(data: Dict[str, any]) -> str:
    """Get analysis from Claude API."""
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": CLAUDE_API_KEY,
    }

    prompt = f"""
    you are an expert in javlin throw and also is excellent in analysing the angles and arm positions etc .
    give us a detailed explanation on :

    Arm Angle Statistics:
    {data['angle_stats']}

    Key Frames:
    {data['key_frames']}

    Based on this data, please provide a detailed report on:
    1. Overall performance assessment
    2. Trends and patterns in arm movements
    3. Specific areas for improvement
    4. Recommendations for enhancing performance
    5. Any potential issues in form or technique suggested by the data

    Please be specific and provide actionable advice. 
    Format your response as a structured report with clear sections and bullet points for easy reading.
    Save the report in output.txt file .
    """

    api_data = {
        "messages": [{"role": "user", "content": prompt}],
        "model": "claude-3-sonnet-20240229",
        "max_tokens": 2000,
    }

    response = requests.post(CLAUDE_API_URL, headers=headers, json=api_data)
    response.raise_for_status()

    return response.json()["content"][0]["text"]

def generate_performance_report(csv_path: str, frame_folder: str) -> str:
    """Generate a comprehensive performance report."""
    df = load_csv_data(csv_path)
    angle_stats = analyze_arm_angles(df)
    plot_paths = create_angle_plots(df)
    key_frames = analyze_key_frames(frame_folder, df)
    
    analysis_data = {
        'angle_stats': angle_stats,
        'key_frames': key_frames
    }
    
    claude_analysis = get_claude_analysis(analysis_data)
    
    report = f"""
    Athlete Performance Analysis Report
    ===================================

    Arm Angle Statistics:
    - Left Arm Mean Angle: {angle_stats['left_mean']:.2f} degrees
    - Left Arm Angle Standard Deviation: {angle_stats['left_std']:.2f} degrees
    - Right Arm Mean Angle: {angle_stats['right_mean']:.2f} degrees
    - Right Arm Angle Standard Deviation: {angle_stats['right_std']:.2f} degrees
    - Arm Symmetry (difference in mean angles): {angle_stats['symmetry']:.2f} degrees

    Key Frames Analysis:
    {', '.join([f"Frame {kf['frame_number']}: Left Angle {kf['left_angle']:.2f}, Right Angle {kf['right_angle']:.2f}" for kf in key_frames])}

    AI Analysis:
    {claude_analysis}

    Note: Please refer to the generated plots and key frame images for visual analysis.
    """
    
    return report

# Example usage
if __name__ == "__main__":
    csv_path = 'path/to/your/output.csv'
    frame_folder = 'path/to/your/extracted_frames'
    report = generate_performance_report(csv_path, frame_folder)
    print(report)
    
    # Optionally, save the report to a file
    with open('performance_report.txt', 'w') as f:
        f.write(report)