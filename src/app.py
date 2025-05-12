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
os.makedirs(REF_DIR, exist_ok=True)
_save_counter = 0  # Simple counter for unique filenames

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
    if emb is None or not refs: return "Unknown", float('inf')
    min_dist = float('inf')
    best_match = "Unknown"
    emb_cpu = emb.cpu()
    
    # Tried cosine similarity first but euclidean distance works better
    for ref in refs:
        dist = torch.nn.functional.pairwise_distance(emb_cpu, ref['embedding'].cpu()).item()
        if dist < min_dist:
            min_dist = dist
            best_match = ref['name']
    return (best_match, min_dist) if min_dist <= thresh else ("Unknown", min_dist)

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
def process_webcam(stop_event, result_q, detector, model, skip_n):
    # FIXME: Sometimes the webcam feed freezes after a few minutes
    # Might be related to threading issues or OpenCV buffer problems
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        result_q.put(("error", "Could not open webcam."))
        return
    frame_count = 0
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.5)
            ret, frame = cap.read() # Try again
            if not ret:
                result_q.put(("error", "Failed to capture frame."))
                break
        frame_count += 1
        if frame_count % (skip_n + 1) == 0:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                boxes, probs = detector.detect(rgb)
                faces = []
                if boxes is not None and probs is not None:
                    for box, prob in zip(boxes, probs):
                        if prob >= DET_THRESH:
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
                                            'box': box, 'prob': prob, 'image': face, 'embedding': emb
                                        })
                result_q.put(("processed_frame", {
                    'frame': cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    'detected_faces': faces
                }))
            except Exception as e:
                print(f"Error processing frame: {e}") # Log error, continue
        time.sleep(0.02) # Prevent busy-waiting
    cap.release()
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
        'latest_faces': [], 'latest_frame': None, 'capture_info': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
            if key == 'refs': print(f"Loaded {len(value)} references from file.")

    # sidebar
    st.sidebar.title("Controls & References")
    current_thresh = st.sidebar.slider("Recognition Threshold", 0.5, 2.0, REC_THRESH, 0.1, 
                                     help="Lower value = stricter matching.")

    if not st.session_state.webcam_active:
        if st.sidebar.button("Start Webcam", key="start"):
            st.session_state.webcam_active = True
            st.session_state.stop_event.clear()
            st.session_state.latest_faces = []
            st.session_state.latest_frame = None
            st.session_state.capture_info = None
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
        if st.sidebar.button("Stop Webcam", key="stop"):
            print("Stop button clicked.")
            st.session_state.stop_event.set()
            st.session_state.webcam_active = False

    st.sidebar.markdown("---")
    st.sidebar.subheader("Add Face from Webcam")
    if st.session_state.webcam_active:
        num_faces = len(st.session_state.latest_faces)
        if num_faces > 0:
            st.sidebar.write(f"Detected {num_faces} face(s).")
            face_ids = list(range(1, num_faces + 1))
            sel_idx = st.sidebar.selectbox("Select face to add:", options=face_ids, index=0)
            idx = sel_idx - 1
            if st.sidebar.button("Capture Selected Face", key="capture"):
                if 0 <= idx < num_faces:
                    face = st.session_state.latest_faces[idx]
                    if face.get('image') is not None and face.get('embedding') is not None:
                        st.session_state.capture_info = {'image': face['image'], 'embedding': face['embedding']}
                        print(f"Captured face {idx+1} for adding.")
                        st.rerun()
                    else: st.sidebar.warning("Selected face data incomplete.")
                else: st.sidebar.error("Invalid face index.")
        else: st.sidebar.info("Point the camera at a face.")
    else: st.sidebar.info("Start webcam to capture faces.")

    if st.session_state.capture_info:
        st.sidebar.markdown("**Confirm Add Face:**")
        try:
            # Convert BGR to RGB for display
            disp_img = cv2.cvtColor(st.session_state.capture_info['image'], cv2.COLOR_BGR2RGB)
            st.sidebar.image(disp_img, caption="Face to Add", width=FACE_SIZE * 2)
        except Exception as e: st.sidebar.error(f"Error displaying captured image: {e}")
        new_name = st.sidebar.text_input("Enter Name:", key="new_face_name").strip()
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("Confirm Add", key="confirm_add"):
                if new_name:
                    st.session_state.refs.append({
                        'name': new_name,
                        'embedding': st.session_state.capture_info['embedding'],
                        'image': st.session_state.capture_info['image']
                    })
                    print(f"Added '{new_name}' to session references.")
                    saved = save_refs(st.session_state.refs)
                    st.sidebar.success(f"Added '{new_name}' and saved references.") if saved else st.sidebar.warning(f"Added '{new_name}' locally, failed save.")
                    st.session_state.capture_info = None
                    st.rerun()
                else: st.sidebar.error("Please enter a name.")
        with col2:
            if st.button("Cancel", key="cancel_add"):
                st.session_state.capture_info = None
                print("Cancelled adding face.")
                st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Known Faces")
    if st.session_state.refs:
        num_refs = len(st.session_state.refs)
        st.sidebar.write(f"{num_refs} reference(s) loaded.")
        # Show faces in a grid - works better on different screen sizes
        cols = 3  # This seems to work well on most displays
        grid = st.sidebar.columns(cols)
        for i, ref in enumerate(st.session_state.refs):
            with grid[i % cols]:
                try:
                    disp_img = cv2.cvtColor(ref['image'], cv2.COLOR_BGR2RGB)
                    st.image(disp_img, caption=ref['name'], width=FACE_SIZE)
                except Exception as e: st.error(f"Err display: {e}")
        if st.sidebar.button("Clear All References", key="clear_all"):
            st.session_state.refs = []
            st.session_state.capture_info = None
            try:
                if os.path.exists(REF_FILE): os.remove(REF_FILE)
                for file in os.listdir(REF_DIR):
                    if file.lower().endswith('.jpg'):
                        try: os.remove(os.path.join(REF_DIR, file))
                        except Exception as e_file: st.warning(f"Could not remove {file}: {e_file}")
                print("Cleared references from session and attempted disk clear.")
            except Exception as e: st.error(f"Error clearing reference files: {e}")
            st.rerun()
    else: st.sidebar.info("No faces added yet.")

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
            elif result_type == "processed_frame":
                # Draw bounding boxes on the frame
                frame = data['frame'].copy()
                faces = data['detected_faces']
                st.session_state.latest_faces = faces
                st.session_state.latest_frame = frame
                recognized = []

                for i, face_data in enumerate(faces):
                    box = face_data['box']
                    prob = face_data['prob']
                    emb = face_data['embedding']
                    x1, y1, x2, y2 = [int(b) for b in box]
                    
                    # Compare with known faces
                    name, dist = compare_faces(emb, st.session_state.refs, current_thresh)

                    # Green for recognized, red for unknown
                    if name != "Unknown":
                        color = (0, 255, 0); label = f"{name} ({dist:.2f})"
                        if name not in recognized: recognized.append(name)
                    else:
                        color = (0, 0, 255); label = f"Unknown #{i+1} (p={prob:.2f})"

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label with background for better visibility
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (x1, y1 - h - 4), (x1 + w, y1), color, -1)
                    cv2.putText(frame, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                frame_place.image(frame, channels="RGB", use_container_width=True)
                
                # Show recognition status
                if recognized: info_place.success(f"Recognized: {', '.join(recognized)}")
                elif faces: info_place.warning(f"Detected {len(faces)} face(s), none recognized.")
                else: info_place.info("No faces detected.")

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