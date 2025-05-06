# app.py
import streamlit as st
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import time
import threading
import os
import pickle
from queue import Queue, Empty
from facenet_pytorch import MTCNN, InceptionResnetV1
import copy
import uuid

# --- Configuration ---
DETECTION_THRESHOLD = 0.9  # Minimum confidence probability for detecting a face
RECOGNITION_THRESHOLD = 1.0  # Maximum distance for considering a match (lower is stricter)
SKIP_FRAMES = 1  # Process every N+1 frames
FACE_IMG_SIZE = 60  # Size to display face crops in sidebar
REFERENCES_DIR = "face_references"  # Directory to store face data
REFERENCES_FILE = os.path.join(REFERENCES_DIR, "face_references.pkl")  # File to store face data

# Ensure references directory exists
os.makedirs(REFERENCES_DIR, exist_ok=True)

# --- Model Loading (Cached) ---
@st.cache_resource
def load_face_detector():
    """Loads the MTCNN face detector."""
    try:
        # Use CUDA if available, otherwise CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        detector = MTCNN(
            keep_all=True,  # Detect all faces
            device=device,
            selection_method='probability'  # Return faces with highest probability first
        )
        st.sidebar.success("Face detector loaded.")
        return detector
    except Exception as e:
        st.error(f"Error loading face detector: {e}")
        return None

@st.cache_resource
def load_recognition_model():
    """Loads the InceptionResnetV1 face recognition model."""
    try:
        # Use CUDA if available, otherwise CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = InceptionResnetV1(pretrained='vggface2', device=device).eval()
        st.sidebar.success("Face recognition model loaded.")
        return model
    except Exception as e:
        st.error(f"Error loading recognition model: {e}")
        return None

# --- Face Processing Functions ---
def detect_faces(frame, face_detector):
    """Detects faces in a frame and returns boxes and probabilities."""
    try:
        # MTCNN expects RGB PIL image or numpy array
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Detect faces
        boxes, probs = face_detector.detect(img_rgb)
        return boxes, probs
    except Exception as e:
        st.sidebar.error(f"Detection error: {e}")
        return None, None

def extract_face_region(frame, box):
    """Extracts the face region from the frame based on the bounding box."""
    x1, y1, x2, y2 = [int(b) for b in box]
    # Ensure coordinates are within image bounds and valid
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 > x1 and y2 > y1:
        return frame[y1:y2, x1:x2]
    return None

def get_face_embedding(face_img, model):
    """Generates an embedding for a single face image."""
    if face_img is None:
        return None
    try:
        # Preprocessing for InceptionResnetV1
        face_img_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        preprocess = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        face_tensor = preprocess(face_img_pil).unsqueeze(0)

        # Move tensor to the model's device
        device = next(model.parameters()).device
        face_tensor = face_tensor.to(device)

        # Generate embedding
        with torch.no_grad():
            embedding = model(face_tensor)
        return embedding
    except Exception as e:
        st.sidebar.error(f"Embedding error: {e}")
        return None

def compare_embedding(embedding, references):
    """Compares a face embedding to a list of reference embeddings."""
    if embedding is None or not references:
        return "Unknown", float('inf')  # Return infinite distance if no references or invalid embedding

    min_dist = float('inf')
    best_match_name = "Unknown"

    for ref in references:
        ref_embedding = ref['embedding']
        # Ensure embeddings are on the same device (CPU for distance calculation)
        dist = torch.nn.functional.pairwise_distance(embedding.cpu(), ref_embedding.cpu()).item()
        if dist < min_dist:
            min_dist = dist
            best_match_name = ref['name']

    # Check against threshold
    if min_dist <= RECOGNITION_THRESHOLD:
        return best_match_name, min_dist
    else:
        return "Unknown", min_dist

# --- File Operations ---
def save_references_to_file(references):
    """Saves face references to a file."""
    try:
        # Save references list to file
        with open(REFERENCES_FILE, 'wb') as f:
            # Store everything except the raw embeddings (which are tensors)
            serializable_refs = []
            for ref in references:
                # Convert tensor to numpy for storage
                embedding_numpy = ref['embedding'].cpu().numpy()
                # Convert image to file
                img_path = os.path.join(REFERENCES_DIR, f"{ref['name']}_{uuid.uuid4().hex[:8]}.jpg")
                cv2.imwrite(img_path, ref['image'])
                
                serializable_refs.append({
                    'name': ref['name'],
                    'embedding_numpy': embedding_numpy,
                    'image_path': img_path
                })
            
            pickle.dump(serializable_refs, f)
        return True
    except Exception as e:
        st.error(f"Error saving references: {e}")
        return False

def load_references_from_file():
    """Loads face references from a file if it exists."""
    if not os.path.exists(REFERENCES_FILE):
        return []
    
    try:
        with open(REFERENCES_FILE, 'rb') as f:
            serializable_refs = pickle.load(f)
            
        # Convert back to proper format with tensors
        references = []
        for ref in serializable_refs:
            # Check if image file exists
            if os.path.exists(ref['image_path']):
                image = cv2.imread(ref['image_path'])
                # Convert numpy back to tensor
                embedding = torch.tensor(ref['embedding_numpy'])
                
                references.append({
                    'name': ref['name'],
                    'embedding': embedding,
                    'image': image
                })
            else:
                st.warning(f"Image for reference '{ref['name']}' not found")
        
        return references
    except Exception as e:
        st.error(f"Error loading references: {e}")
        return []

# --- Webcam Processing Thread ---
def process_webcam_feed(stop_event, result_queue, face_detector, model, skip_frames):
    """Continuously processes webcam frames in a separate thread."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        result_queue.put(("error", "Could not open webcam."))
        return

    frame_count = 0
    last_processed_data = None

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            result_queue.put(("error", "Failed to capture frame."))
            break

        frame_count += 1
        current_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process every Nth frame or if no data processed yet
        if frame_count % (skip_frames + 1) == 0 or last_processed_data is None:
            boxes, probs = detect_faces(frame, face_detector)
            detected_faces_data = []

            if boxes is not None and probs is not None:
                for box, prob in zip(boxes, probs):
                    if prob >= DETECTION_THRESHOLD:
                        face_img = extract_face_region(frame, box)
                        if face_img is not None:
                            embedding = get_face_embedding(face_img, model)
                            detected_faces_data.append({
                                'box': box,
                                'prob': prob,
                                'image': face_img,  # Store the BGR image crop
                                'embedding': embedding
                            })

            # Store processed data
            last_processed_data = {
                'frame': current_frame_rgb,  # Send RGB frame for display
                'detected_faces': detected_faces_data
            }
            result_queue.put(("processed_frame", last_processed_data))
        else:
            # Send unprocessed frame for smoother display
            result_queue.put(("raw_frame", current_frame_rgb))

        time.sleep(0.01)  # Small delay

    cap.release()
    result_queue.put(("stopped", None))

# --- Streamlit App UI ---
def main():
    st.set_page_config(layout="wide", page_title="Face Recognition Demo")
    st.title("Face Detection & Recognition Demo")
    st.write("Shows face detection bounding boxes and allows adding faces for recognition.")

    # --- Initialization ---
    if 'references_initialized' not in st.session_state:
        # First run: load existing references from file
        st.session_state.references = load_references_from_file()
        st.session_state.references_initialized = True
        
    if 'webcam_active' not in st.session_state:
        st.session_state.webcam_active = False
    if 'stop_event' not in st.session_state:
        st.session_state.stop_event = threading.Event()
    if 'result_queue' not in st.session_state:
        st.session_state.result_queue = Queue()
    if 'latest_detected_faces' not in st.session_state:
        st.session_state.latest_detected_faces = []
    if 'latest_frame' not in st.session_state:
        st.session_state.latest_frame = None
    if 'capture_info' not in st.session_state:
        # Stores info about the face selected for adding
        st.session_state.capture_info = None  # {'image': np.array, 'embedding': tensor}

    # Load models
    face_detector = load_face_detector()
    model = load_recognition_model()

    if face_detector is None or model is None:
        st.error("Failed to load models. Cannot start.")
        return

    # --- Sidebar ---
    st.sidebar.title("Controls & References")

    # Show current recognition threshold
    current_threshold = st.sidebar.slider(
        "Recognition Threshold", 
        min_value=0.5, 
        max_value=2.0, 
        value=RECOGNITION_THRESHOLD,
        step=0.1,
        help="Lower value = stricter matching. Increase if faces aren't being recognized."
    )
    
    # Webcam Controls
    if not st.session_state.webcam_active:
        if st.sidebar.button("Start Webcam", key="start"):
            st.session_state.webcam_active = True
            st.session_state.stop_event.clear()
            st.session_state.latest_detected_faces = []  # Clear old detections
            st.session_state.latest_frame = None
            st.session_state.capture_info = None

            # Clear the queue
            while not st.session_state.result_queue.empty():
                try:
                    st.session_state.result_queue.get_nowait()
                except Empty:
                    break

            # Start processing thread
            thread = threading.Thread(
                target=process_webcam_feed,
                args=(
                    st.session_state.stop_event,
                    st.session_state.result_queue,
                    face_detector,
                    model,
                    SKIP_FRAMES
                ),
                daemon=True
            )
            thread.start()
            st.rerun()  # Rerun to enter the display loop

    else:  # Webcam is active
        if st.sidebar.button("Stop Webcam", key="stop"):
            st.session_state.stop_event.set()
            st.session_state.webcam_active = False
            # Don't rerun immediately, let the loop handle the stop signal

    st.sidebar.markdown("---")

    # Add Face Section
    st.sidebar.subheader("Add Face from Webcam")
    if st.session_state.webcam_active and st.session_state.latest_detected_faces:
        num_detected = len(st.session_state.latest_detected_faces)
        st.sidebar.write(f"Detected {num_detected} face(s).")

        # Button to capture the *first* detected face for adding
        if st.session_state.latest_detected_faces:
            idx = st.sidebar.number_input(
                "Face index to add", 
                min_value=1, 
                max_value=len(st.session_state.latest_detected_faces), 
                value=1, 
                step=1
            ) - 1

            if st.sidebar.button("Capture Selected Face", key="capture"):
                face_to_add = st.session_state.latest_detected_faces[idx]
                if face_to_add['embedding'] is not None:
                    st.session_state.capture_info = {
                        'image': face_to_add['image'],
                        'embedding': face_to_add['embedding']
                    }
                    st.rerun()
                else:
                    st.sidebar.warning("Could not get embedding for the selected face.")

    elif st.session_state.webcam_active:
        st.sidebar.info("Point the camera at a face.")

    # Display captured face and ask for name
    if st.session_state.capture_info:
        st.sidebar.markdown("**Confirm Add Face:**")
        # Display BGR image correctly with cv2 conversion for display
        st.sidebar.image(cv2.cvtColor(st.session_state.capture_info['image'], cv2.COLOR_BGR2RGB),
                        caption="Face to Add", width=FACE_IMG_SIZE * 2)
        
        new_name = st.sidebar.text_input("Enter Name:", key="new_face_name")

        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("Confirm Add", key="confirm_add"):
                if new_name and st.session_state.capture_info['embedding'] is not None:
                    # Always append so you can collect multiple images per person
                    st.session_state.references.append({
                        'name': new_name,
                        'embedding': st.session_state.capture_info['embedding'],
                        'image': st.session_state.capture_info['image']
                    })
                    if save_references_to_file(st.session_state.references):
                        st.sidebar.success(f"Added another image for '{new_name}' and saved to disk!")
                    else:
                        st.sidebar.warning(f"Added locally for '{new_name}', but failed to save to disk.")
                    st.session_state.capture_info = None
                    st.rerun()
                elif not new_name:
                    st.sidebar.error("Please enter a name.")
                else:
                    st.sidebar.error("Cannot add face, embedding missing.")

        with col2:
            if st.button("Cancel", key="cancel_add"):
                st.session_state.capture_info = None
                st.rerun()

    st.sidebar.markdown("---")

    # Display References
    st.sidebar.subheader("Known Faces")
    if st.session_state.references:
        # Use columns for better layout
        num_cols = 3
        cols = st.sidebar.columns(num_cols)
        for i, ref in enumerate(st.session_state.references):
            with cols[i % num_cols]:
                # Display BGR image correctly
                st.image(cv2.cvtColor(ref['image'], cv2.COLOR_BGR2RGB),
                        caption=ref['name'], width=FACE_IMG_SIZE)
        
        if st.sidebar.button("Clear All References", key="clear_all"):
            st.session_state.references = []
            st.session_state.capture_info = None  # Clear any pending capture
            # Remove all files in the references directory
            for file in os.listdir(REFERENCES_DIR):
                if file.endswith('.jpg') or file == 'face_references.pkl':
                    os.remove(os.path.join(REFERENCES_DIR, file))
            st.rerun()
    else:
        st.sidebar.info("No faces added yet.")

    # --- Debug Info ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Debug Info")
    st.sidebar.write(f"Loaded {len(st.session_state.references)} reference faces")
    if st.sidebar.button("Reload References from Disk"):
        st.session_state.references = load_references_from_file()
        st.rerun()

    # --- Main Area ---
    frame_placeholder = st.empty()
    info_placeholder = st.empty()

    if not st.session_state.webcam_active and not st.session_state.references:
        st.info("ℹ️ Start the webcam and point it at a face. Use the sidebar controls to capture and add faces for recognition.")
    elif not st.session_state.webcam_active and st.session_state.references:
        st.info("ℹ️ Start the webcam to begin recognition.")

    # --- Display Loop (runs when webcam is active) ---
    while st.session_state.webcam_active:
        try:
            # Get data from the queue
            result_type, data = st.session_state.result_queue.get(timeout=0.1)  # Short timeout

            if result_type == "error":
                st.error(data)
                st.session_state.webcam_active = False
                st.session_state.stop_event.set()
                st.rerun()
                break
            elif result_type == "stopped":
                st.session_state.webcam_active = False
                st.info("Webcam stopped.")
                st.rerun()
                break
            elif result_type == "raw_frame":
                # Display unprocessed frame if available (smoother video)
                if st.session_state.latest_frame is None:  # Only update if no processed frame yet
                    frame_placeholder.image(data, channels="RGB", use_container_width=True)
                    st.session_state.latest_frame = data  # Keep track of the latest raw frame

            elif result_type == "processed_frame":
                annotated_frame = data['frame'].copy()  # Work on a copy
                detected_faces = data['detected_faces']
                st.session_state.latest_detected_faces = copy.deepcopy(detected_faces)  # Store for adding
                st.session_state.latest_frame = annotated_frame  # Update latest frame

                recognition_summary = []

                # Draw boxes and recognize
                for i, face_data in enumerate(detected_faces):
                    box = face_data['box']
                    prob = face_data['prob']
                    embedding = face_data['embedding']
                    x1, y1, x2, y2 = [int(b) for b in box]

                    name, distance = compare_embedding(embedding, st.session_state.references)

                    if distance <= current_threshold:
                        color = (0,255,0)
                        label = f"{name} ({distance:.2f})"
                    else:
                        color = (0,0,255)
                        # show index + probability so you know which “Unknown” this is
                        label = f"Unknown #{i+1} (p={prob:.2f})"

                    # draw box + label exactly the same
                    cv2.rectangle(annotated_frame, (x1,y1),(x2,y2), color, 2)
                    (w,h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .5, 2)
                    cv2.rectangle(annotated_frame, (x1, y1 - h - 4),(x1+w, y1), color, -1)
                    cv2.putText(annotated_frame, label, (x1, y1 - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 2)

                # Update main frame display
                frame_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)

                # Update info text
                if recognition_summary:
                    info_placeholder.success(f"Recognized: {', '.join(recognition_summary)}")
                elif detected_faces:
                    info_placeholder.warning(f"Detected {len(detected_faces)} face(s), but none recognized.")
                else:
                    info_placeholder.info("No faces detected.")

        except Empty:
            # If queue is empty, display the last known frame to avoid flickering
            if st.session_state.latest_frame is not None:
                frame_placeholder.image(st.session_state.latest_frame, channels="RGB", use_container_width=True)
            time.sleep(0.05)  # Wait briefly before checking queue again
            continue  # Continue loop to check stop_event and queue again

        except Exception as e:
            st.error(f"Error in display loop: {e}")
            st.session_state.webcam_active = False
            st.session_state.stop_event.set()
            st.rerun()
            break

    # Cleanup when loop finishes (e.g., webcam stopped)
    if not st.session_state.webcam_active:
        # Ensure thread resources are potentially cleaned up if needed
        # (Daemon threads usually exit automatically, but good practice)
        st.session_state.stop_event.set()


if __name__ == "__main__":
    main()