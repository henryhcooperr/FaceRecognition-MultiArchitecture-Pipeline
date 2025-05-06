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

# Configs 
DETECTION_THRESHOLD = 0.9
RECOGNITION_THRESHOLD = 1.0
SKIP_FRAMES = 1
FACE_IMG_SIZE = 60
REFERENCES_DIR = "face_references"
REFERENCES_FILE = os.path.join(REFERENCES_DIR, "face_references.pkl")
os.makedirs(REFERENCES_DIR, exist_ok=True)
_reference_save_counter = 0 # Simple counter for unique filenames

# face processing fuinctions

def get_face_embedding(face_img, model):
    if face_img is None or face_img.size == 0: return None
    try:
        face_img_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        preprocess = transforms.Compose([
            transforms.Resize((160, 160)), transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        face_tensor = preprocess(face_img_pil).unsqueeze(0).to(next(model.parameters()).device)
        with torch.no_grad(): embedding = model(face_tensor)
        return embedding
    except Exception as e:
        st.sidebar.error(f"Embedding error: {e}")
        return None

def compare_embedding(embedding, references, threshold):
    if embedding is None or not references: return "Unknown", float('inf')
    min_dist = float('inf')
    best_match_name = "Unknown"
    embedding_cpu = embedding.cpu()
    for ref in references:
        dist = torch.nn.functional.pairwise_distance(embedding_cpu, ref['embedding'].cpu()).item()
        if dist < min_dist:
            min_dist = dist
            best_match_name = ref['name']
    return (best_match_name, min_dist) if min_dist <= threshold else ("Unknown", min_dist)

# file operations
def save_references_to_file(references):
    global _reference_save_counter
    try:
        serializable_refs = []
        for ref in references:
            _reference_save_counter += 1
            img_filename = f"{ref['name'].replace(' ', '_')}_{_reference_save_counter}.jpg"
            img_path = os.path.join(REFERENCES_DIR, img_filename)
            if cv2.imwrite(img_path, ref['image']):
                serializable_refs.append({
                    'name': ref['name'],
                    'embedding_numpy': ref['embedding'].cpu().numpy(),
                    'image_path': img_path
                })
            else:
                 st.warning(f"Failed to save image file: {img_path}")
                 continue # Skip this entry if image save failed
        with open(REFERENCES_FILE, 'wb') as f: pickle.dump(serializable_refs, f)
        return True
    except Exception as e:
        st.error(f"Error saving references: {e}")
        return False

def load_references_from_file():
    if not os.path.exists(REFERENCES_FILE): return []
    references = []
    try:
        with open(REFERENCES_FILE, 'rb') as f: serializable_refs = pickle.load(f)
        for ref_data in serializable_refs:
            if os.path.exists(ref_data['image_path']):
                image = cv2.imread(ref_data['image_path'])
                if image is not None:
                    references.append({
                        'name': ref_data['name'],
                        'embedding': torch.tensor(ref_data['embedding_numpy']).cpu(),
                        'image': image
                    })
                else: st.warning(f"Could not load image for '{ref_data['name']}' from {ref_data['image_path']}")
            else: st.warning(f"Image file not found for '{ref_data['name']}': {ref_data['image_path']}")
        return references
    except Exception as e:
        st.error(f"Error loading references: {e}")
        return []

# webcam processing 
def process_webcam_feed(stop_event, result_queue, face_detector, model, skip_frames):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        result_queue.put(("error", "Could not open webcam."))
        return
    frame_count = 0
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.5)
            ret, frame = cap.read() # Try again
            if not ret:
                result_queue.put(("error", "Failed to capture frame."))
                break
        frame_count += 1
        if frame_count % (skip_frames + 1) == 0:
            try:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                boxes, probs = face_detector.detect(img_rgb)
                detected_faces_data = []
                if boxes is not None and probs is not None:
                    for box, prob in zip(boxes, probs):
                        if prob >= DETECTION_THRESHOLD:
                            x1, y1, x2, y2 = [int(b) for b in box]
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                            if x2 > x1 and y2 > y1:
                                face_img = frame[y1:y2, x1:x2]
                                if face_img.size > 0:
                                    embedding = get_face_embedding(face_img, model)
                                    if embedding is not None:
                                        detected_faces_data.append({
                                            'box': box, 'prob': prob, 'image': face_img, 'embedding': embedding
                                        })
                result_queue.put(("processed_frame", {
                    'frame': cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    'detected_faces': detected_faces_data
                }))
            except Exception as e:
                print(f"Error during frame processing: {e}") # Log error, continue
        time.sleep(0.02) # Prevent busy-waiting
    cap.release()
    result_queue.put(("stopped", None))

# streamlit app ui
def main():
    st.set_page_config(layout="wide", page_title="Face Recognition Demo")
    st.title("Face Detection & Recognition Demo")
    st.write("Shows face detection bounding boxes and allows adding faces for recognition.")

    @st.cache_resource
    def load_models():
        print("Loading models...")
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            detector = MTCNN(keep_all=True, device=device, selection_method='probability')
            model = InceptionResnetV1(pretrained='vggface2', device=device).eval()
            print(f"Models loaded on device: {device}")
            return detector, model
        except Exception as e:
            st.error(f"Fatal error loading models: {e}")
            return None, None

    face_detector, model = load_models()
    if face_detector is None or model is None:
        st.error("Failed to load models. Cannot start the application.")
        st.stop()

    # Initialize session state
    defaults = {
        'references': load_references_from_file(), 'webcam_active': False,
        'stop_event': threading.Event(), 'result_queue': Queue(),
        'latest_detected_faces': [], 'latest_frame_rgb': None, 'capture_info': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
            if key == 'references': print(f"Loaded {len(value)} references from file.")

    # sidebar
    st.sidebar.title("Controls & References")
    current_threshold = st.sidebar.slider("Recognition Threshold", 0.5, 2.0, RECOGNITION_THRESHOLD, 0.1, help="Lower value = stricter matching.")

    if not st.session_state.webcam_active:
        if st.sidebar.button("Start Webcam", key="start"):
            st.session_state.webcam_active = True
            st.session_state.stop_event.clear()
            st.session_state.latest_detected_faces = []
            st.session_state.latest_frame_rgb = None
            st.session_state.capture_info = None
            while not st.session_state.result_queue.empty():
                try: st.session_state.result_queue.get_nowait()
                except Empty: break
            threading.Thread(
                target=process_webcam_feed,
                args=(st.session_state.stop_event, st.session_state.result_queue, face_detector, model, SKIP_FRAMES),
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
        num_detected = len(st.session_state.latest_detected_faces)
        if num_detected > 0:
            st.sidebar.write(f"Detected {num_detected} face(s).")
            face_indices = list(range(1, num_detected + 1))
            selected_index = st.sidebar.selectbox("Select face index to add:", options=face_indices, index=0)
            idx = selected_index - 1
            if st.sidebar.button("Capture Selected Face", key="capture"):
                if 0 <= idx < num_detected:
                    face_to_add = st.session_state.latest_detected_faces[idx]
                    if face_to_add.get('image') is not None and face_to_add.get('embedding') is not None:
                        st.session_state.capture_info = {'image': face_to_add['image'], 'embedding': face_to_add['embedding']}
                        print(f"Captured face {idx+1} for adding.")
                        st.rerun()
                    else: st.sidebar.warning("Selected face data incomplete.")
                else: st.sidebar.error("Invalid face index.")
        else: st.sidebar.info("Point the camera at a face.")
    else: st.sidebar.info("Start webcam to capture faces.")

    if st.session_state.capture_info:
        st.sidebar.markdown("**Confirm Add Face:**")
        try:
            display_img = cv2.cvtColor(st.session_state.capture_info['image'], cv2.COLOR_BGR2RGB)
            st.sidebar.image(display_img, caption="Face to Add", width=FACE_IMG_SIZE * 2)
        except Exception as e: st.sidebar.error(f"Error displaying captured image: {e}")
        new_name = st.sidebar.text_input("Enter Name:", key="new_face_name").strip()
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("Confirm Add", key="confirm_add"):
                if new_name:
                    st.session_state.references.append({
                        'name': new_name,
                        'embedding': st.session_state.capture_info['embedding'],
                        'image': st.session_state.capture_info['image']
                    })
                    print(f"Added '{new_name}' to session references.")
                    save_status = save_references_to_file(st.session_state.references)
                    st.sidebar.success(f"Added '{new_name}' and saved references.") if save_status else st.sidebar.warning(f"Added '{new_name}' locally, failed save.")
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
    if st.session_state.references:
        num_refs = len(st.session_state.references)
        st.sidebar.write(f"{num_refs} reference(s) loaded.")
        num_cols = 3
        cols = st.sidebar.columns(num_cols)
        for i, ref in enumerate(st.session_state.references):
            with cols[i % num_cols]:
                try:
                    ref_display_img = cv2.cvtColor(ref['image'], cv2.COLOR_BGR2RGB)
                    st.image(ref_display_img, caption=ref['name'], width=FACE_IMG_SIZE)
                except Exception as e: st.error(f"Err dsp ref {i}: {e}")
        if st.sidebar.button("Clear All References", key="clear_all"):
            st.session_state.references = []
            st.session_state.capture_info = None
            try:
                if os.path.exists(REFERENCES_FILE): os.remove(REFERENCES_FILE)
                for file in os.listdir(REFERENCES_DIR):
                    if file.lower().endswith('.jpg'):
                        try: os.remove(os.path.join(REFERENCES_DIR, file))
                        except Exception as e_rem: st.warning(f"Could not remove {file}: {e_rem}")
                print("Cleared references from session and attempted disk clear.")
            except Exception as e: st.error(f"Error clearing reference files: {e}")
            st.rerun()
    else: st.sidebar.info("No faces added yet.")

    # Main Area 
    frame_placeholder = st.empty()
    info_placeholder = st.empty()

    if not st.session_state.webcam_active:
        info_msg = "ℹ️ Start webcam & point at face. Use sidebar to add faces." if not st.session_state.references else "ℹ️ Start webcam for recognition."
        info_placeholder.info(info_msg)
        if st.session_state.latest_frame_rgb is not None:
             frame_placeholder.image(st.session_state.latest_frame_rgb, channels="RGB", use_container_width=True)
        else:
             frame_placeholder.markdown("<div style='height: 480px; border: 1px dashed gray; display: flex; justify-content: center; align-items: center;'>Webcam Off</div>", unsafe_allow_html=True)

    # display loop
    while st.session_state.webcam_active:
        try:
            result_type, data = st.session_state.result_queue.get(timeout=0.1)

            if result_type == "error":
                st.error(f"Webcam Error: {data}")
                st.session_state.webcam_active = False
                st.session_state.stop_event.set()
                st.rerun(); break
            elif result_type == "stopped":
                print("Received stopped signal from thread.")
                st.session_state.webcam_active = False
                info_placeholder.info("Webcam stopped.")
                if st.session_state.latest_frame_rgb is not None:
                    frame_placeholder.image(st.session_state.latest_frame_rgb, channels="RGB", use_container_width=True)
                st.rerun(); break
            elif result_type == "processed_frame":
                annotated_frame = data['frame'].copy()
                detected_faces = data['detected_faces']
                st.session_state.latest_detected_faces = detected_faces
                st.session_state.latest_frame_rgb = annotated_frame
                recognition_summary = []

                for i, face_data in enumerate(detected_faces):
                    box = face_data['box']
                    prob = face_data['prob']
                    embedding = face_data['embedding']
                    x1, y1, x2, y2 = [int(b) for b in box]
                    name, distance = compare_embedding(embedding, st.session_state.references, current_threshold)

                    if name != "Unknown":
                        color = (0, 255, 0); label = f"{name} ({distance:.2f})"
                        if name not in recognition_summary: recognition_summary.append(name)
                    else:
                        color = (0, 0, 255); label = f"Unknown #{i+1} (p={prob:.2f})"

                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(annotated_frame, (x1, y1 - h - 4), (x1 + w, y1), color, -1)
                    cv2.putText(annotated_frame, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                frame_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)
                if recognition_summary: info_placeholder.success(f"Recognized: {', '.join(recognition_summary)}")
                elif detected_faces: info_placeholder.warning(f"Detected {len(detected_faces)} face(s), none recognized.")
                else: info_placeholder.info("No faces detected.")

        except Empty:
            if st.session_state.latest_frame_rgb is not None:
                frame_placeholder.image(st.session_state.latest_frame_rgb, channels="RGB", use_container_width=True)
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

if __name__ == "__main__":
    main()