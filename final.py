import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
import gradio as gr
import time

# Global variables for frame-by-frame navigation
current_frame_index = 0
video1_cap = None
video2_cap = None

# Function to process frames in blocks of 5000 pixels
def process_in_blocks(frame1, frame2, block_area=5000, threshold=40):
    if frame1.shape != frame2.shape:
        frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))

    # Resize frames to grayscale for easier comparison
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculate the block size (height × width of a block)
    height, width = gray1.shape
    total_pixels = height * width
    block_size = max(1, int(np.sqrt(block_area)))  # Find block size dimension

    # Initialize overlay for bounding boxes
    overlay = np.zeros_like(frame1)

    # Similarity/dissimilarity counters
    similar_blocks = 0
    dissimilar_blocks = 0

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            y_end = min(y + block_size, height)
            x_end = min(x + block_size, width)

            # Extract blocks from both frames
            block1 = gray1[y:y_end, x:x_end]
            block2 = gray2[y:y_end, x:x_end]

            # Calculate average pixel intensity difference
            mean_diff = np.abs(block1.astype(np.int32) - block2.astype(np.int32)).mean()

            # Determine if the block is similar or dissimilar
            if mean_diff <= threshold:
                similar_blocks += 1
                color = (0, 255, 0)  # Green for similar
            else:
                dissimilar_blocks += 1
                color = (0, 0, 255)  # Red for dissimilar

            # Draw bounding boxes on the overlay
            cv2.rectangle(overlay, (x, y), (x_end, y_end), color, -1)

    # Calculate similarity and dissimilarity percentages
    total_blocks = similar_blocks + dissimilar_blocks
    similarity_percentage = round((similar_blocks / total_blocks) * 100, 2)
    dissimilarity_percentage = round((dissimilar_blocks / total_blocks) * 100, 2)

    # Combine frames side-by-side
    combined_frame = np.concatenate((frame1, frame2), axis=1)
    combined_overlay = np.concatenate((overlay, overlay), axis=1)
    result_frame = cv2.addWeighted(combined_frame, 0.7, combined_overlay, 0.3, 0)

    return combined_frame, result_frame, similarity_percentage, dissimilarity_percentage

# Initialize video captures
def initialize_videos(video1_path, video2_path):
    global video1_cap, video2_cap, current_frame_index
    video1_cap = cv2.VideoCapture(video1_path)
    video2_cap = cv2.VideoCapture(video2_path)
    current_frame_index = 0

# Compare the next frame and return similarity/dissimilarity
def compare_next_frame():
    global video1_cap, video2_cap, current_frame_index

    if not video1_cap.isOpened() or not video2_cap.isOpened():
        return "Error: Videos are not loaded.", None, None, None, None

    #  Read the next frame from each video
    
    ret1, frame1 = video1_cap.read()
    ret2, frame2 = video2_cap.read()

    # End of one or both videos
    if not ret1 or not ret2:
        return (
            "End of one or both videos reached.",
            None,
            None,
            None,
            None,
        )

    # Compare the current frame using block-based logic
    block_area = 10000  # Each block contains approximately 10000 pixels
    threshold = 40  # Define pixel difference threshold
    raw_frame, result_frame, similarity_score, dissimilarity_score = process_in_blocks(
        frame1, frame2, block_area, threshold
    )

    # Increment frame index
    current_frame_index += 1

    # Convert frames to RGB for Gradio
    raw_frame_rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
    result_frame_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)

    # Wait for 500ms before returning the results
    time.sleep(0.5)

    return (
        f"Frame {current_frame_index}: Comparison completed.",
        similarity_score,
        dissimilarity_score,
        raw_frame_rgb,
        result_frame_rgb,
    )

# Gradio UI
def initialize_and_process_videos(video1, video2):
    # Initialize the videos for frame-by-frame navigation
    initialize_videos(video1, video2)
    return compare_next_frame()

# Gradio Interface
interface = gr.Blocks()

with interface:
    gr.Markdown("# Video Comparison Tool with Block-Based Pixel Grouping")
    gr.Markdown(
        "Upload two videos to compare them frame by frame. "
        "The comparison is performed in blocks of approximately 5000 pixels, "
        "with similarity/dissimilarity scores displayed along with the frames."
    )

    with gr.Row():
        video1_input = gr.Video(label="Upload Video 1")
        video2_input = gr.Video(label="Upload Video 2")

    compare_button = gr.Button("Compare & Load Videos")
    status_output = gr.Textbox(label="Status")
    similarity_output = gr.Number(label="Similarity (%)")
    dissimilarity_output = gr.Number(label="Dissimilarity (%)")
    raw_frames_output = gr.Image(label="Side-by-Side Raw Frames")
    video_frames_output = gr.Image(label="Side-by-Side Frames with Bounding Boxes")

    next_frame_button = gr.Button("Next Frame")

    # Compare and load videos
    compare_button.click(
        initialize_and_process_videos,
        inputs=[video1_input, video2_input],
        outputs=[
            status_output,
            similarity_output,
            dissimilarity_output,
            raw_frames_output,
            video_frames_output,
        ],
    )

    # Process next frame
    next_frame_button.click(
        compare_next_frame,
        inputs=[],
        outputs=[
            status_output,
            similarity_output,
            dissimilarity_output,
            raw_frames_output,
            video_frames_output,
        ],
    )

# Launch Gradio app
if __name__ == "__main__":
    interface.launch(share=True)
