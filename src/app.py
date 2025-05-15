# app.py - Streamlit interface for real-time face recognition
# Started: 4/10/2025 - Actually got it working: 4/16/2025 (many late nights)
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

# Configs and constants - tweaked through trial and error
# Bumped detection threshold from 0.8 to 0.9 to reduce false positives
DET_THRESH = 0.9  
# Initially tried 0.7 but got too many false matches - 1.0 seems like a good balance
REC_THRESH = 1.0  
SKIP_FRAMES = 1  # Processing every frame was too slow on my laptop
FACE_SIZE = 60
REF_DIR = "face_references"
REF_FILE = os.path.join(REF_DIR, "face_references.pkl")
HISTORY_FILE = os.path.join(REF_DIR, "recognition_history.pkl")
os.makedirs(REF_DIR, exist_ok=True)
_save_counter = 0  # Simple counter for unique filenames
# Face tracking constants
TRACKING_THRESHOLD = 0.3  # IOU threshold for face tracking between frames

# Face processing functions
def get_embedding(face_img, model):
    """Extract face embedding using the model"""
    if face_img is None or face_img.size == 0: return None
    try:
        # Had issues with color format conversion - be explicit about BGR to RGB
        pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        # These normalization values work best with the VGGFace2 model
        preprocess = transforms.Compose([
            transforms.Resize((160, 160)), transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        face_tensor = preprocess(pil_img).unsqueeze(0).to(next(model.parameters()).device)
        with torch.no_grad(): emb = model(face_tensor)
        return emb
    except Exception as e:
        st.sidebar.error(f"Embedding error: {e}")
        return None

def compare_faces(emb, refs, thresh):
    if emb is None or not refs: return "Unknown", float('inf'), None
    min_dist = float('inf')
    best_match = "Unknown"
    best_ref_idx = None
    emb_cpu = emb.cpu()
    
    # Tried cosine similarity first but euclidean distance works better
    for i, ref in enumerate(refs):
        dist = torch.nn.functional.pairwise_distance(emb_cpu, ref['embedding'].cpu()).item()
        if dist < min_dist:
            min_dist = dist
            best_match = ref['name']
            best_ref_idx = i
    return (best_match, min_dist, best_ref_idx) if min_dist <= thresh else ("Unknown", min_dist, None)

# file operations
def save_refs(refs):
    """Save reference faces to disk - had to debug this a lot"""
    global _save_counter
    try:
        saveable_refs = []
        for ref in refs:
            _save_counter += 1
            # Actually better to create unique names with counter
            # Fixed bug where same-named people would overwrite each other
            img_file = f"{ref['name'].replace(' ', '_')}_{_save_counter:08x}.jpg"
            img_path = os.path.join(REF_DIR, img_file)
            if cv2.imwrite(img_path, ref['image']):
                saveable_refs.append({
                    'name': ref['name'],
                    'embedding_numpy': ref['embedding'].cpu().numpy(),
                    'image_path': img_path
                })
            else:
                 st.warning(f"Failed to save image: {img_path}")
                 continue # Skip this entry if image save failed
        with open(REF_FILE, 'wb') as f: pickle.dump(saveable_refs, f)
        return True
    except Exception as e:
        st.error(f"Error saving references: {e}")
        return False

# Old version - don't use, had issues with paths
# def save_refs_v1(refs):
#     """Save reference faces to disk."""
#     try:
#         with open(REF_FILE, 'wb') as f:
#             pickle.dump(refs, f)
#         return True
#     except Exception as e:
#         st.error(f"Error saving references: {e}")
#         return False

def load_refs():
    if not os.path.exists(REF_FILE): return []
    refs = []
    try:
        with open(REF_FILE, 'rb') as f: saved_refs = pickle.load(f)
        for ref in saved_refs:
            if os.path.exists(ref['image_path']):
                img = cv2.imread(ref['image_path'])
                if img is not None:
                    refs.append({
                        'name': ref['name'],
                        'embedding': torch.tensor(ref['embedding_numpy']).cpu(),
                        'image': img
                    })
                else: st.warning(f"Could not load image for '{ref['name']}' from {ref['image_path']}")
            else: st.warning(f"Image file missing for '{ref['name']}': {ref['image_path']}")
        return refs
    except Exception as e:
        st.error(f"Error loading references: {e}")
        return []

# webcam processing 
def calc_iou(box1, box2):
    """Calculate Intersection over Union between two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection area
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0  # No intersection
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0

def process_webcam(stop_event, result_q, detector, model, skip_n):
    # FIXME: Sometimes the webcam feed freezes after a few minutes
    # Might be related to threading issues or OpenCV buffer problems
    cap = cv2.VideoCapture(0)
    # Attempt to set higher resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        result_q.put(("error", "Could not open webcam."))
        return
    
    frame_count = 0
    # Track faces between frames
    prev_boxes = []
    face_ids = []
    face_id_counter = 0
    
    while not stop_event.is_set():
        # Try/except around the entire loop to prevent crashes
        try:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.5)
                ret, frame = cap.read() # Try again
                if not ret:
                    result_q.put(("error", "Failed to capture frame."))
                    break
                    
            frame_count += 1
            if frame_count % (skip_n + 1) == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                boxes, probs = detector.detect(rgb)
                faces = []
                matched_prev_boxes = [False] * len(prev_boxes)
                
                if boxes is not None and probs is not None:
                    # Match current faces with previous faces using IOU
                    current_ids = [None] * len(boxes)
                    
                    # First pass: match faces with previous boxes
                    for i, (box, prob) in enumerate(zip(boxes, probs)):
                        if prob < DET_THRESH:
                            continue
                            
                        x1, y1, x2, y2 = [int(b) for b in box]
                        # Noticed some negative coordinates in rare cases
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                        
                        if x2 <= x1 or y2 <= y1:
                            continue  # Invalid box
                            
                        # Try to match with a previous box
                        max_iou = 0
                        max_idx = -1
                        for j, prev_box in enumerate(prev_boxes):
                            if matched_prev_boxes[j]:
                                continue  # Already matched
                                
                            iou = calc_iou(box, prev_box)
                            if iou > max_iou and iou > TRACKING_THRESHOLD:
                                max_iou = iou
                                max_idx = j
                        
                        # If matched, keep the same ID
                        if max_idx >= 0:
                            current_ids[i] = face_ids[max_idx]
                            matched_prev_boxes[max_idx] = True
                        else:
                            # New face detected
                            current_ids[i] = face_id_counter
                            face_id_counter += 1
                    
                    # Second pass: process all faces
                    for i, (box, prob) in enumerate(zip(boxes, probs)):
                        if prob < DET_THRESH or current_ids[i] is None:
                            continue
                            
                        x1, y1, x2, y2 = [int(b) for b in box]
                        # Noticed some negative coordinates in rare cases
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                        
                        if x2 > x1 and y2 > y1:
                            face = frame[y1:y2, x1:x2]
                            if face.size > 0:
                                emb = get_embedding(face, model)
                                if emb is not None:
                                    faces.append({
                                        'box': box, 'prob': prob, 'image': face, 
                                        'embedding': emb, 'face_id': current_ids[i]
                                    })
                
                # Update for next frame
                if boxes is not None and len(boxes) > 0:
                    prev_boxes = [box.tolist() for box, prob in zip(boxes, probs) if prob >= DET_THRESH]
                    face_ids = [id for id in current_ids if id is not None]
                    result_q.put(("face_tracking", {'face_id_counter': face_id_counter}))
                
                result_q.put(("processed_frame", {
                    'frame': cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    'detected_faces': faces
                }))
            time.sleep(0.02) # Prevent busy-waiting
        except Exception as e:
            print(f"Error in webcam processing loop: {e}") # Log error, continue
            time.sleep(0.5)  # Pause briefly before continuing
    
    # Release camera when stopping
    try:
        cap.release()
    except Exception as e:
        print(f"Error releasing camera: {e}")
    
    result_q.put(("stopped", None))

# streamlit app ui
def main():
    st.set_page_config(layout="wide", page_title="Face Recognition Demo")
    st.title("Face Detection & Recognition Demo")
    st.write("Shows face detection bounding boxes and allows adding faces for recognition.")

    # This was a huge optimization - cache the models so they don't reload every time
    # I spent hours trying to figure out why the app was so slow before this
    @st.cache_resource
    def load_models():
        print("Loading models...")
        try:
            # Check for GPU - helped speed things up a lot on my NVIDIA laptop
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            detector = MTCNN(keep_all=True, device=device, selection_method='probability')
            model = InceptionResnetV1(pretrained='vggface2', device=device).eval()
            print(f"Models loaded on device: {device}")
            return detector, model
        except Exception as e:
            st.error(f"Fatal error loading models: {e}")
            return None, None

    detector, model = load_models()
    if detector is None or model is None:
        st.error("Failed to load models. Cannot start the application.")
        st.stop()

    # Initialize session state
    defaults = {
        'refs': load_refs(), 'webcam_active': False,
        'stop_event': threading.Event(), 'result_q': Queue(),
        'latest_faces': [], 'latest_frame': None, 'capture_info': None,
        'recognition_history': [], 'prev_face_boxes': [], 'face_id_counter': 0,
        'tracked_faces': {}, 'show_history': False, 'edit_ref_idx': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
            if key == 'refs': print(f"Loaded {len(value)} references from file.")

    # sidebar
    st.sidebar.title("Controls & References")
    
    # Create tabs for different sidebar sections
    tab_controls, tab_faces, tab_history = st.sidebar.tabs(["🎛️ Controls", "👤 Faces", "📜 History"])
    
    # Controls tab
    with tab_controls:
        current_thresh = st.slider("Recognition Threshold", 0.5, 2.0, REC_THRESH, 0.1, 
                                    help="Lower value = stricter matching.")

    # Webcam controls in the Controls tab
    with tab_controls:
        if not st.session_state.webcam_active:
            if st.button("🎥 Start Webcam", key="start", use_container_width=True):
                st.session_state.webcam_active = True
                st.session_state.stop_event.clear()
                st.session_state.latest_faces = []
                st.session_state.latest_frame = None
                st.session_state.capture_info = None
                st.session_state.tracked_faces = {}
                # Clear out any pending messages in the queue
                while not st.session_state.result_q.empty():
                    try: st.session_state.result_q.get_nowait()
                    except Empty: break
                threading.Thread(
                    target=process_webcam,
                    args=(st.session_state.stop_event, st.session_state.result_q, detector, model, SKIP_FRAMES),
                    daemon=True
                ).start()
                print("Webcam thread started.")
                st.rerun()
        else:
            if st.button("⏹️ Stop Webcam", key="stop", use_container_width=True):
                print("Stop button clicked.")
                st.session_state.stop_event.set()
                st.session_state.webcam_active = False

    # Face selection in the Faces tab
    with tab_faces:
        # Only show manual face selection if we're not already capturing a face and not editing
        if not st.session_state.capture_info and st.session_state.edit_ref_idx is None:
            st.subheader("Select a Face to Add")
            if st.session_state.webcam_active:
                num_faces = len(st.session_state.latest_faces)
                if num_faces > 0:
                    st.write(f"Detected {num_faces} face(s) - select any face below:")
                    
                    # Initialize selected face index if not already in session state
                    if 'selected_face_idx' not in st.session_state:
                        st.session_state.selected_face_idx = 0
                    
                    # Display face thumbnails in a grid
                    cols = min(3, num_faces)
                    face_cols = st.columns(cols)
                    
                    for i, face_data in enumerate(st.session_state.latest_faces):
                        face_img = face_data['image']
                        if face_img is not None:
                            with face_cols[i % cols]:
                                # Convert BGR to RGB for display
                                rgb_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                                face_id = face_data.get('face_id', i)
                                st.image(rgb_img, width=70, caption=f"Face #{face_id}")
                                if st.button(f"Select #{face_id}", key=f"sel_face_{i}", use_container_width=True):
                                    st.session_state.selected_face_idx = i
                                    # Force UI refresh when selection changes
                                    st.rerun()
                    
                    # Highlight and display currently selected face
                    st.markdown("---")
                    if 0 <= st.session_state.selected_face_idx < num_faces:
                        selected_face = st.session_state.latest_faces[st.session_state.selected_face_idx]
                        selected_img = selected_face['image']
                        if selected_img is not None:
                            st.markdown(f"**Selected Face #{selected_face.get('face_id', st.session_state.selected_face_idx + 1)}:**")
                            st.image(cv2.cvtColor(selected_img, cv2.COLOR_BGR2RGB), width=150)
                            
                            if st.button("➕ Add This Person", key="capture", use_container_width=True):
                                if selected_face.get('image') is not None and selected_face.get('embedding') is not None:
                                    st.session_state.capture_info = {
                                        'image': selected_face['image'], 
                                        'embedding': selected_face['embedding']
                                    }
                                    print(f"Captured face {st.session_state.selected_face_idx+1} for adding.")
                                    st.rerun()
                                else: 
                                    st.warning("Selected face data incomplete.")
                        else:
                            st.error("Selected face image is invalid.")
                else: 
                    st.info("🔍 Point the camera at a face.")
            else: 
                st.info("▶️ Start webcam from the Controls tab to detect faces.")
        # Otherwise, just show a note about the current action
        elif st.session_state.webcam_active and st.session_state.capture_info:
            st.info("Enter a name for the detected face 👇")

    # Face capture and editing UI - in Faces tab
    with tab_faces:
        # Face capture UI
        if st.session_state.capture_info:
            st.markdown("### 👤 Add This Face")
            st.markdown("**Enter a name for this person:**")
            try:
                # Convert BGR to RGB for display
                disp_img = cv2.cvtColor(st.session_state.capture_info['image'], cv2.COLOR_BGR2RGB)
                st.image(disp_img, caption="Face to Add", width=FACE_SIZE * 3)
            except Exception as e: st.error(f"Error displaying captured image: {e}")
            
            # Make the text input more prominent
            new_name = st.text_input(
                "👋 Person Name:", 
                key="new_face_name", 
                placeholder="Enter name here...",
                help="Enter this person's name to add them to the recognition system"
            ).strip()
            
            cols = st.columns(2)
            # Use a more prominent button layout
            if new_name:
                if cols[0].button("✅ Save as '" + new_name + "'", key="confirm_add", use_container_width=True):
                    st.session_state.refs.append({
                        'name': new_name,
                        'embedding': st.session_state.capture_info['embedding'],
                        'image': st.session_state.capture_info['image'],
                        'timestamp': time.time()
                    })
                    print(f"Added '{new_name}' to session references.")
                    saved = save_refs(st.session_state.refs)
                    st.success(f"Added '{new_name}' and saved references.") if saved else st.warning(f"Added '{new_name}' locally, failed save.")
                    
                    # Add to history
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.recognition_history.append({
                        'name': new_name,
                        'action': 'added',
                        'timestamp': timestamp
                    })
                    
                    st.session_state.capture_info = None
                    st.rerun()
            else:
                st.warning("👆 Please enter a name above")
                
            if cols[1].button("❌ Skip This Face", key="cancel_add", use_container_width=True):
                st.session_state.capture_info = None
                print("Cancelled adding face.")
                st.rerun()
        
        # Face editing UI
        elif st.session_state.edit_ref_idx is not None and 0 <= st.session_state.edit_ref_idx < len(st.session_state.refs):
            ref = st.session_state.refs[st.session_state.edit_ref_idx]
            st.markdown("### ✏️ Edit Person")
            try:
                # Convert BGR to RGB for display
                disp_img = cv2.cvtColor(ref['image'], cv2.COLOR_BGR2RGB)
                st.image(disp_img, caption="Edit Person", width=FACE_SIZE * 3)
            except Exception as e: st.error(f"Error displaying reference image: {e}")
            
            # Make the text input more prominent
            new_name = st.text_input(
                "✏️ Update Name:", 
                value=ref['name'],
                key="edit_face_name", 
                help="Edit this person's name"
            ).strip()
            
            cols = st.columns(3)
            # Save button
            if new_name:
                if cols[0].button("✅ Save Changes", key="confirm_edit", use_container_width=True):
                    old_name = ref['name']
                    ref['name'] = new_name
                    saved = save_refs(st.session_state.refs)
                    
                    # Add to history
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.recognition_history.append({
                        'name': new_name,
                        'old_name': old_name if old_name != new_name else None,
                        'action': 'edited',
                        'timestamp': timestamp
                    })
                    
                    st.success(f"Updated to '{new_name}' and saved references.") if saved else st.warning(f"Updated locally, failed save.")
                    st.session_state.edit_ref_idx = None
                    st.rerun()
            else:
                st.warning("Please enter a name")
                
            # Delete button
            if cols[1].button("🗑️ Delete Person", key="delete_ref", use_container_width=True):
                deleted_name = ref['name']
                st.session_state.refs.pop(st.session_state.edit_ref_idx)
                saved = save_refs(st.session_state.refs)
                
                # Add to history
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.recognition_history.append({
                    'name': deleted_name,
                    'action': 'deleted',
                    'timestamp': timestamp
                })
                
                st.success(f"Deleted '{deleted_name}' and saved references.") if saved else st.warning(f"Deleted locally, failed save.")
                st.session_state.edit_ref_idx = None
                st.rerun()
                
            # Cancel button
            if cols[2].button("❌ Cancel", key="cancel_edit", use_container_width=True):
                st.session_state.edit_ref_idx = None
                st.rerun()

    # Known faces display - in Faces tab    
    with tab_faces:
        st.markdown("---")
        st.subheader("Known Faces")
        if st.session_state.refs:
            num_refs = len(st.session_state.refs)
            st.write(f"{num_refs} reference(s) loaded.")
            # Show faces in a grid - works better on different screen sizes
            cols = 4  # More columns for better layout
            grid = st.columns(cols)
            for i, ref in enumerate(st.session_state.refs):
                with grid[i % cols]:
                    try:
                        disp_img = cv2.cvtColor(ref['image'], cv2.COLOR_BGR2RGB)
                        st.image(disp_img, caption=ref['name'], width=FACE_SIZE)
                        # Add edit button below each face
                        if st.button("✏️", key=f"edit_ref_{i}", help=f"Edit {ref['name']}"):
                            st.session_state.edit_ref_idx = i
                            st.rerun()
                    except Exception as e: st.error(f"Err display: {e}")
            
            st.markdown("---")
            if st.button("Clear All References", key="clear_all"):
                st.session_state.refs = []
                st.session_state.capture_info = None
                st.session_state.edit_ref_idx = None
                try:
                    if os.path.exists(REF_FILE): os.remove(REF_FILE)
                    for file in os.listdir(REF_DIR):
                        if file.lower().endswith('.jpg'):
                            try: os.remove(os.path.join(REF_DIR, file))
                            except Exception as e_file: st.warning(f"Could not remove {file}: {e_file}")
                    print("Cleared references from session and attempted disk clear.")
                    
                    # Add to history
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.recognition_history.append({
                        'action': 'cleared_all',
                        'timestamp': timestamp
                    })
                    
                except Exception as e: st.error(f"Error clearing reference files: {e}")
                st.rerun()
        else: st.info("No faces added yet. Use the webcam to add faces.")
    
    # Recognition history tab
    with tab_history:
        st.subheader("Recognition History")
        if st.session_state.recognition_history:
            for item in reversed(st.session_state.recognition_history[:30]):  # Show last 30 items
                if item['action'] == 'recognized':
                    st.success(f"🔍 {item['timestamp']}: Recognized **{item['name']}**")
                elif item['action'] == 'added':
                    st.info(f"➕ {item['timestamp']}: Added **{item['name']}** to references")
                elif item['action'] == 'edited':
                    if item.get('old_name') and item['old_name'] != item['name']:
                        st.warning(f"✏️ {item['timestamp']}: Renamed **{item['old_name']}** to **{item['name']}**")
                    else:
                        st.warning(f"✏️ {item['timestamp']}: Edited **{item['name']}**")
                elif item['action'] == 'deleted':
                    st.error(f"🗑️ {item['timestamp']}: Deleted **{item['name']}**")
                elif item['action'] == 'cleared_all':
                    st.error(f"🧹 {item['timestamp']}: Cleared all references")
            
            if st.button("Clear History", key="clear_history"):
                st.session_state.recognition_history = []
                st.rerun()
        else:
            st.info("No recognition history yet. Start the webcam to begin.")

    # Main Area 
    frame_place = st.empty()
    info_place = st.empty()

    if not st.session_state.webcam_active:
        info_msg = "ℹ️ Start webcam & point at face. Use sidebar to add faces." if not st.session_state.refs else "ℹ️ Start webcam for recognition."
        info_place.info(info_msg)
        if st.session_state.latest_frame is not None:
             frame_place.image(st.session_state.latest_frame, channels="RGB", use_container_width=True)
        else:
             frame_place.markdown("<div style='height: 480px; border: 1px dashed gray; display: flex; justify-content: center; align-items: center;'>Webcam Off</div>", unsafe_allow_html=True)

    # display loop
    while st.session_state.webcam_active:
        try:
            result_type, data = st.session_state.result_q.get(timeout=0.1)

            if result_type == "error":
                st.error(f"Webcam Error: {data}")
                st.session_state.webcam_active = False
                st.session_state.stop_event.set()
                st.rerun(); break
            elif result_type == "stopped":
                print("Received stopped signal from thread.")
                st.session_state.webcam_active = False
                info_place.info("Webcam stopped.")
                if st.session_state.latest_frame is not None:
                    frame_place.image(st.session_state.latest_frame, channels="RGB", use_container_width=True)
                st.rerun(); break
            elif result_type == "face_tracking":
                # Update face_id_counter from the processing thread
                st.session_state.face_id_counter = data['face_id_counter']
            elif result_type == "processed_frame":
                # Draw bounding boxes on the frame
                frame = data['frame'].copy()
                faces = data['detected_faces']
                st.session_state.latest_faces = faces
                st.session_state.latest_frame = frame
                recognized = []

                tracked_faces = {}
                for i, face_data in enumerate(faces):
                    box = face_data['box']
                    prob = face_data['prob']
                    emb = face_data['embedding']
                    face_id = face_data.get('face_id', i)
                    x1, y1, x2, y2 = [int(b) for b in box]
                    
                    # Compare with known faces
                    name, dist, ref_idx = compare_faces(emb, st.session_state.refs, current_thresh)
                    
                    # Store tracked face data
                    tracked_faces[face_id] = {
                        'box': box, 
                        'name': name, 
                        'dist': dist,
                        'last_seen': time.time()
                    }

                    # Green for recognized, red for unknown
                    if name != "Unknown":
                        color = (255, 165, 0); label = f"{name} ({dist:.2f})"  # Orange for recognized
                        if name not in recognized: 
                            recognized.append(name)
                            # Add to history if newly recognized
                            if not any(h.get('name') == name and h.get('action') == 'recognized' 
                                       and time.time() - time.mktime(time.strptime(h.get('timestamp', ''), "%Y-%m-%d %H:%M:%S")) < 60 
                                       for h in st.session_state.recognition_history[-10:] if h.get('action') == 'recognized'):
                                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                                st.session_state.recognition_history.append({
                                    'name': name,
                                    'action': 'recognized',
                                    'timestamp': timestamp
                                })
                    else:
                        color = (0, 0, 255); label = f"Unknown #{face_id} (p={prob:.2f})"  # Red for unknown

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label with black background for better visibility
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1 - h - 8), (x1 + w + 4, y1), (0, 0, 0), -1)
                    cv2.putText(frame, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Update the tracked faces dictionary
                st.session_state.tracked_faces = tracked_faces

                frame_place.image(frame, channels="RGB", use_container_width=True)
                
                # Show recognition status and auto-prompt for unrecognized faces
                if recognized: 
                    info_place.success(f"Recognized: {', '.join(recognized)}")
                elif faces:
                    # Auto-select first unrecognized face if none are currently captured
                    if not st.session_state.capture_info and len(faces) > 0:
                        st.session_state.selected_face_idx = 0
                        st.session_state.capture_info = {
                            'image': faces[0]['image'], 
                            'embedding': faces[0]['embedding']
                        }
                        info_place.warning(f"Detected {len(faces)} unrecognized face(s). Please enter a name in the sidebar!")
                        # Force a rerun to update the UI with the capture dialog
                        st.rerun()
                    else:
                        info_place.warning(f"Detected {len(faces)} face(s), none recognized. Add a name in the sidebar.")
                else: 
                    info_place.info("No faces detected.")

        except Empty:
            # No new frame yet, show last frame if available
            if st.session_state.latest_frame is not None:
                frame_place.image(st.session_state.latest_frame, channels="RGB", use_container_width=True)
            time.sleep(0.05) # Prevent high CPU usage when queue is empty
            continue
        except Exception as e:
            st.error(f"Error in display loop: {e}")
            st.session_state.webcam_active = False
            st.session_state.stop_event.set()
            st.rerun(); break

    if not st.session_state.webcam_active:
        st.session_state.stop_event.set() # Ensure stop event is set
        print("Display loop finished.")

# Old version of main loop - keeping for reference
# def main_v1():
#     st.title("Face Recognition")
#     if st.button("Start"):
#         cam = cv2.VideoCapture(0)
#         frame_placeholder = st.empty()
#         while True:
#             ret, frame = cam.read()
#             if not ret:
#                 st.error("Failed to get frame")
#                 break
#             frame_placeholder.image(frame, channels="BGR")
#             if st.button("Stop"):
#                 break
#         cam.release()

if __name__ == "__main__":
    main()