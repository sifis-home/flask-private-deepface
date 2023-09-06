import json
import os
import time

import cv2
import numpy as np
import rel
import websocket

from deepface import DeepFace
from deepface.commons import functions

# dependency configuration
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def on_error(ws, error):
    print(error)


def on_close(ws, close_status_code, close_msg):
    print("### Connection closed ###")


def on_open(ws):
    print("### Connection established ###")


ws = websocket.WebSocketApp(
    "ws://localhost:3000/ws",
    on_open=on_open,
    on_error=on_error,
    on_close=on_close,
)

ws.run_forever(dispatcher=rel)  # Set dispatcher to automatic reconnection
rel.signal(2, rel.abort)  # Keyboard Interrupt

divisible = 60


def private_analysis(
    db_path,
    model_name="VGG-Face",
    detector_backend="opencv",
    distance_metric="cosine",
    enable_face_analysis=False,
    source=0,
    privacy_parameter=9,
    time_threshold=5,
    frame_threshold=5,
):
    # global variables
    text_color = (255, 255, 255)
    pivot_img_size = 112  # face recognition result image

    enable_emotion = False
    enable_age_gender = False
    # ------------------------
    # find custom values for this input set
    target_size = functions.find_target_size(model_name=model_name)
    # ------------------------
    # build models once to store them in the memory
    # otherwise, they will be built after cam started and this will cause delays
    DeepFace.build_model(model_name=model_name)
    print(model_name)
    print(f"facial recognition model {model_name} is just built")

    # -----------------------
    # call a dummy find function for db_path once to create embeddings in the initialization
    DeepFace.find(
        img_path=np.zeros([224, 224, 3]),
        db_path=db_path,
        model_name=model_name,
        detector_backend=detector_backend,
        distance_metric=distance_metric,
        enforce_detection=False,
    )
    print("Called DeepFace.find")
    # -----------------------
    # visualization
    freeze = False
    face_detected = False
    face_included_frames = (
        0  # freeze screen if face detected sequantially 5 frames
    )
    freezed_frame = 0
    tic = time.time()

    cap = cv2.VideoCapture(source)  # webcam
    frame_id = 0
    labels_dict = {}

    while True:
        _, img = cap.read()
        if isinstance(img, np.ndarray) == True:
            raw_img = img.copy()
            resolution = img.shape
            resolution_x = img.shape[1]
            resolution_y = img.shape[0]
            img = cv2.GaussianBlur(
                img, (int(privacy_parameter), int(privacy_parameter)), 0
            )

        if img is None:
            # Image loading failed
            print("Failed to load the image.")
            break

        if freeze == False:
            try:
                # just extract the regions to highlight in webcam
                face_objs = DeepFace.extract_faces(
                    img_path=img,
                    target_size=target_size,
                    detector_backend=detector_backend,
                    enforce_detection=False,
                )
                faces = []
                for face_obj in face_objs:
                    facial_area = face_obj["facial_area"]
                    faces.append(
                        (
                            facial_area["x"],
                            facial_area["y"],
                            facial_area["w"],
                            facial_area["h"],
                        )
                    )
            except:  # to avoid exception if no face detected
                faces = []

            if len(faces) == 0:
                face_included_frames = 0
        else:
            faces = []

        detected_faces = []
        face_index = 0
        for x, y, w, h in faces:
            if w > 130:  # discard small detected faces
                face_detected = True
                if face_index == 0:
                    face_included_frames = (
                        face_included_frames + 1
                    )  # increase frame for a single face

                detected_face = img[
                    int(y) : int(y + h), int(x) : int(x + w)
                ]  # crop detected face

                # -------------------------------------

                detected_faces.append((x, y, w, h))
                face_index = face_index + 1

                # -------------------------------------

        if (
            face_detected == True
            and face_included_frames == frame_threshold
            and freeze == False
        ):
            freeze = True
            # base_img = img.copy()
            base_img = raw_img.copy()
            detected_faces_final = detected_faces.copy()
            tic = time.time()

        if freeze == True:
            # print("frame_id3:", frame_id)
            toc = time.time()
            if (toc - tic) < time_threshold:
                if freezed_frame == 0:
                    freeze_img = base_img.copy()

                    labels = []

                    for detected_face in detected_faces_final:
                        x = detected_face[0]
                        y = detected_face[1]
                        w = detected_face[2]
                        h = detected_face[3]

                        # -------------------------------
                        # extract detected face
                        custom_face = base_img[y : y + h, x : x + w]
                        # -------------------------------

                        dfs = DeepFace.find(
                            img_path=custom_face,
                            db_path=db_path,
                            model_name=model_name,
                            detector_backend=detector_backend,
                            distance_metric=distance_metric,
                            enforce_detection=False,
                            silent=True,
                        )

                        if len(dfs) > 0:
                            # directly access 1st item because custom face is extracted already
                            df = dfs[0]

                            if df.shape[0] > 0:
                                candidate = df.iloc[0]
                                label = candidate["identity"]
                                labels.append(label)

                                # to use this source image as private
                                display_img = cv2.imread(label)
                                display_img = cv2.cvtColor(
                                    cv2.GaussianBlur(
                                        display_img,
                                        (
                                            int(privacy_parameter),
                                            int(privacy_parameter),
                                        ),
                                        0,
                                    ),
                                    cv2.COLOR_BGR2RGB,
                                )

                        tic = (
                            time.time()
                        )  # in this way, freezed image can show 5 seconds

                        # -------------------------------
                    for label in labels:
                        if label not in labels_dict.values():
                            labels_dict[frame_id] = label

                time_left = int(time_threshold - (toc - tic) + 1)

                freezed_frame = freezed_frame + 1
            else:
                face_detected = False
                face_included_frames = 0
                freeze = False
                freezed_frame = 0

        else:
            # cv2.imshow("img", img)
            labels = []

        if cv2.waitKey(1) & 0xFF == ord("q"):  # press q to quit
            break

        for label in labels:
            if label not in labels_dict.values():
                labels_dict[frame_id] = label

        if frame_id % divisible == 0:
            print("Frame ID: ", frame_id, " Labels: ", labels)

            ws_req = {
                "RequestPostTopicUUID": {
                    "topic_name": "SIFIS:Privacy_Aware_Face_Recognition_Frame_Results",
                    "topic_uuid": "Face_Recognition_Frame_Results",
                    "value": {
                        "description": "Face Recognition Frame Results",
                        "Type": "Video_file",
                        "file_name": str(source),
                        "Privacy_Parameter": int(privacy_parameter),
                        "Frame": int(frame_id),
                        "Labels": labels,
                        "length": int(len(labels)),
                    },
                }
            }
            ws.send(json.dumps(ws_req))

        frame_id += 1
    # kill open cv things
    cap.release()
    cv2.destroyAllWindows()
    ws_req_final = {
        "RequestPostTopicUUID": {
            "topic_name": "SIFIS:Privacy_Aware_Face_Recognition_Results",
            "topic_uuid": "Face_Recognition_Results",
            "value": {
                "description": "Face Recognition Results",
                "Type": "Video_file",
                "file_name": str(source),
                "Privacy_Parameter": int(privacy_parameter),
                "Labels": labels_dict,
                "count_frames": int(frame_id),
            },
        }
    }
    ws.send(json.dumps(ws_req_final))
    return labels_dict
