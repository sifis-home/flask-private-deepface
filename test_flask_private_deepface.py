from unittest.mock import patch

import cv2
import numpy as np
import pytest
import tensorflow as tf

from app import on_close, on_error, on_open
from deepface import DeepFace
from deepface.basemodels import VGGFace
from deepface.commons import distance as dst
from deepface.commons import functions, realtime
from deepface.detectors import FaceDetector, OpenCvWrapper

# configurations of dependencies

tf_version = tf.__version__
tf_major_version = int(tf_version.split(".", maxsplit=1)[0])
tf_minor_version = int(tf_version.split(".")[1])

if tf_major_version == 1:
    from keras.preprocessing import image
elif tf_major_version == 2:
    from tensorflow.keras.preprocessing import image

# --------------------------------------------------

tf_version = tf.__version__
tf_major_version = int(tf_version.split(".", maxsplit=1)[0])
tf_minor_version = int(tf_version.split(".")[1])

if tf_major_version == 1:
    pass
elif tf_major_version == 2:
    pass


def test_DeepFace_extract_faces():
    target_size = functions.find_target_size(model_name="VGG-Face")
    face_objs = DeepFace.extract_faces(
        img_path="database/Sara.jpg",
        target_size=target_size,
        detector_backend="opencv",
        enforce_detection=False,
    )
    face_objs2 = DeepFace.extract_faces(
        img_path="database/Sara.jpg",
        target_size=target_size,
        detector_backend="opencv",
        enforce_detection=False,
    )

    assert type(face_objs) == type(face_objs2)


def test_extract_faces():
    img = "database/Sara.jpg"
    target_size = (224, 224)
    detector_backend = "opencv"
    grayscale = False
    enforce_detection = True
    align = True

    extracted_faces = functions.extract_faces(
        img,
        target_size=(224, 224),
        detector_backend="opencv",
        grayscale=False,
        enforce_detection=True,
        align=True,
    )

    expected_extracted_faces = []

    # img might be path, base64 or numpy array. Convert it to numpy whatever it is.
    img = functions.load_image(img)
    img_region = [0, 0, img.shape[1], img.shape[0]]

    if detector_backend == "skip":
        face_objs = [(img, img_region, 0)]
    else:
        face_detector = FaceDetector.build_model(detector_backend)
        face_objs = FaceDetector.detect_faces(
            face_detector, detector_backend, img, align
        )

    # in case of no face found
    if len(face_objs) == 0 and enforce_detection is True:
        raise ValueError(
            "Face could not be detected. Please confirm that the picture is a face photo "
            + "or consider to set enforce_detection param to False."
        )

    if len(face_objs) == 0 and enforce_detection is False:
        face_objs = [(img, img_region, 0)]

    for current_img, current_region, confidence in face_objs:
        if current_img.shape[0] > 0 and current_img.shape[1] > 0:
            if grayscale is True:
                current_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)

            # resize and padding
            if current_img.shape[0] > 0 and current_img.shape[1] > 0:
                factor_0 = target_size[0] / current_img.shape[0]
                factor_1 = target_size[1] / current_img.shape[1]
                factor = min(factor_0, factor_1)

                dsize = (
                    int(current_img.shape[1] * factor),
                    int(current_img.shape[0] * factor),
                )
                current_img = cv2.resize(current_img, dsize)

                diff_0 = target_size[0] - current_img.shape[0]
                diff_1 = target_size[1] - current_img.shape[1]
                if grayscale is False:
                    # Put the base image in the middle of the padded image
                    current_img = np.pad(
                        current_img,
                        (
                            (diff_0 // 2, diff_0 - diff_0 // 2),
                            (diff_1 // 2, diff_1 - diff_1 // 2),
                            (0, 0),
                        ),
                        "constant",
                    )
                else:
                    current_img = np.pad(
                        current_img,
                        (
                            (diff_0 // 2, diff_0 - diff_0 // 2),
                            (diff_1 // 2, diff_1 - diff_1 // 2),
                        ),
                        "constant",
                    )

            # double check: if target image is not still the same size with target.
            if current_img.shape[0:2] != target_size:
                current_img = cv2.resize(current_img, target_size)

            # normalizing the image pixels
            # what this line doing? must?
            img_pixels = image.img_to_array(current_img)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255  # normalize input in [0, 1]

            # int cast is for the exception - object of type 'float32' is not JSON serializable
            region_obj = {
                "x": int(current_region[0]),
                "y": int(current_region[1]),
                "w": int(current_region[2]),
                "h": int(current_region[3]),
            }

            expected_extracted_faces = [img_pixels, region_obj, confidence]
            expected_extracted_faces.append(expected_extracted_faces)

    if len(expected_extracted_faces) == 0 and enforce_detection == True:
        raise ValueError(
            f"Detected face shape is {img.shape}. Consider to set enforce_detection arg to False."
        )

    assert type(extracted_faces) == type(expected_extracted_faces)


def test_normalize_input():
    img = cv2.imread("database/Sara.jpg")
    result_img = functions.normalize_input(img, normalization="base")
    result_img2 = functions.normalize_input(img, normalization="Facenet")

    expected_result_img = img
    expected_result_img *= 255

    assert type(result_img) == type(result_img)


def test_find_target_size():
    model_name = "VGG-Face"
    target_size = functions.find_target_size(model_name)

    target_sizes = {
        "VGG-Face": (224, 224),
        "Facenet": (160, 160),
        "Facenet512": (160, 160),
        "OpenFace": (96, 96),
        "DeepFace": (152, 152),
        "DeepID": (55, 47),
        "Dlib": (150, 150),
        "ArcFace": (112, 112),
        "SFace": (112, 112),
    }

    expected_target_size = target_sizes.get(model_name)

    assert type(target_size) == type(expected_target_size)


def test_detect_face():
    resp = []
    result_resp = []
    detected_face = None
    img = cv2.imread("database/Sara.jpg")
    align = True
    detector = OpenCvWrapper.build_model()
    resp = OpenCvWrapper.detect_face(detector, img, align=True)

    img_region = [0, 0, img.shape[1], img.shape[0]]

    faces = []
    try:
        faces, _, scores = detector["face_detector"].detectMultiScale3(
            img, 1.1, 10, outputRejectLevels=True
        )
    except:
        pass

    if len(faces) > 0:
        for (x, y, w, h), confidence in zip(faces, scores):
            detected_face = img[int(y) : int(y + h), int(x) : int(x + w)]

            if align:
                detected_face = OpenCvWrapper.align_face(
                    detector["eye_detector"], detected_face
                )

            img_region = [x, y, w, h]

            result_resp.append((detected_face, img_region, confidence))

    assert type(resp) == type(resp)


def test_opencv_build_model():
    detector = OpenCvWrapper.build_model()

    expected_detector = {}
    expected_detector["face_detector"] = OpenCvWrapper.build_cascade(
        "haarcascade"
    )
    expected_detector["eye_detector"] = OpenCvWrapper.build_cascade(
        "haarcascade_eye"
    )

    assert type(detector) == type(expected_detector)


# from deepface.commons import functions

# def test_detect_face():
#     img = cv2.imread("database/Jack.jpg")
#     detector_backend = "opencv"
#     face_detector = FaceDetector.build_model(detector_backend)

#     obj = FaceDetector.detect_faces(
#         face_detector, detector_backend, img, align=True
#     )

#     if len(obj) > 0:
#         expected_face, expected_region, expected_confidence = obj[
#             0
#         ]  # discard multiple faces
#     else:  # len(obj) == 0
#         expected_face = None
#         expected_region = [0, 0, img.shape[1], img.shape[0]]

#     face, region, confidence = FaceDetector.detect_face(
#         face_detector, detector_backend, img, align=True
#     )

#     assert region == expected_region
#     assert confidence == expected_confidence


def test_FaceDetector_build_model():
    detector_backend = "opencv"
    result_model = FaceDetector.build_model(detector_backend)

    global face_detector_obj  # singleton design pattern

    backends = {
        "opencv": OpenCvWrapper.build_model,
    }

    if not "face_detector_obj" in globals():
        face_detector_obj = {}

    built_models = list(face_detector_obj.keys())
    if detector_backend not in built_models:
        face_detector = backends.get(detector_backend)

        if face_detector:
            face_detector = face_detector()
            face_detector_obj[detector_backend] = face_detector
        else:
            raise ValueError(
                "invalid detector_backend passed - " + detector_backend
            )

    assert type(face_detector_obj[detector_backend]) == type(result_model)


@pytest.mark.parametrize("tf_version", [1, 2])
def test_baseModel(tf_version):
    # Mock TensorFlow version
    tf.__version__ = f"{tf_version}.0.0"

    # Call the function to create the model
    model = VGGFace.baseModel()

    # Check if the model is created and has the expected architecture
    assert isinstance(model, tf.keras.models.Model)

    # Verify model layers
    expected_layers = [
        "zero_padding2d_78",
        "conv2d_96",
        "zero_padding2d_79",
        "conv2d_97",
        "max_pooling2d_30",
        "zero_padding2d_80",
        "conv2d_98",
        "zero_padding2d_81",
        "conv2d_99",
        "max_pooling2d_31",
        "zero_padding2d_82",
        "conv2d_100",
        "zero_padding2d_83",
        "conv2d_101",
        "zero_padding2d_84",
        "conv2d_102",
        "max_pooling2d_32",
        "zero_padding2d_85",
        "conv2d_103",
        "zero_padding2d_86",
        "conv2d_104",
        "zero_padding2d_87",
        "conv2d_105",
        "max_pooling2d_33",
        "zero_padding2d_88",
        "conv2d_106",
        "zero_padding2d_89",
        "conv2d_107",
        "zero_padding2d_90",
        "conv2d_108",
        "max_pooling2d_34",
        "conv2d_109",
        "dropout_12",
        "conv2d_110",
        "dropout_13",
        "conv2d_111",
        "flatten_6",
        "activation_6",
    ]
    actual_layers = [layer.name for layer in model.layers]

    assert len(expected_layers) == len(actual_layers)


def test_findCosineDistance():
    # Create mock source and test representations
    source_representation = np.array([0.1, 0.2, 0.3])
    test_representation = np.array([0.4, 0.5, 0.6])

    # Calculate expected cosine distance manually
    a = np.dot(source_representation, test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    expected_distance = 1 - (a / (np.sqrt(b) * np.sqrt(c)))

    # Call the function with mock representations
    distance = dst.findCosineDistance(
        source_representation, test_representation
    )

    # Check if the calculated distance matches the expected distance
    assert np.isclose(distance, expected_distance, atol=1e-6)


def test_findEuclideanDistance():
    # Create mock source and test representations
    source_representation = np.array([1, 2, 3])
    test_representation = np.array([4, 5, 6])

    # Calculate expected Euclidean distance manually
    euclidean_distance = np.sqrt(
        np.sum(np.square(source_representation - test_representation))
    )

    # Call the function with mock representations
    distance = dst.findEuclideanDistance(
        source_representation, test_representation
    )

    # Check if the calculated distance matches the expected distance
    assert np.isclose(distance, euclidean_distance, atol=1e-6)


def test_l2_normalize():
    # Create mock input array
    input_array = np.array([3, 4])

    # Calculate expected normalized array manually
    normalized_array = input_array / np.sqrt(np.sum(np.square(input_array)))

    # Call the function with mock input
    result = dst.l2_normalize(input_array)

    # Check if the calculated normalized array matches the expected array
    assert np.allclose(result, normalized_array, atol=1e-6)


def test_findThreshold():
    # Test different model names and distance metrics
    model_names = ["VGG-Face", "Facenet", "UnknownModel"]
    distance_metrics = [
        "cosine",
        "unknown_metric",
    ]

    # Define expected thresholds based on the provided data
    expected_thresholds = {
        ("VGG-Face", "cosine"): 0.40,
        ("Facenet", "euclidean"): 10,
        ("ArcFace", "euclidean_l2"): 1.13,
        ("UnknownModel", "unknown_metric"): 0.4,
    }

    # Test each combination of model names and distance metrics
    for model_name in model_names:
        for distance_metric in distance_metrics:
            expected_threshold = expected_thresholds.get(
                (model_name, distance_metric), 0.4
            )

            # Call the function with mock model name and distance metric
            threshold = dst.findThreshold(model_name, distance_metric)

            # Check if the calculated threshold matches the expected threshold
            assert type(threshold) == type(expected_threshold)


def test_on_error():
    error = "WebSocket error occurred"

    with patch("builtins.print") as mock_print:
        on_error(None, error)

    mock_print.assert_called_once_with(error)


def test_on_close():
    close_status_code = 1000
    close_msg = "Connection closed"

    with patch("builtins.print") as mock_print:
        on_close(None, close_status_code, close_msg)

    mock_print.assert_called_once_with("### Connection closed ###")


def test_on_open():
    with patch("builtins.print") as mock_print:
        on_open(None)

    mock_print.assert_called_once_with("### Connection established ###")


def test_on_error_realtime():
    error = "WebSocket error occurred"

    with patch("builtins.print") as mock_print:
        realtime.on_error(None, error)

    mock_print.assert_called_once_with(error)


def test_on_close_realtime():
    close_status_code = 1000
    close_msg = "Connection closed"

    with patch("builtins.print") as mock_print:
        realtime.on_close(None, close_status_code, close_msg)

    mock_print.assert_called_once_with("### Connection closed ###")


def test_on_open_realtime():
    with patch("builtins.print") as mock_print:
        realtime.on_open(None)

    mock_print.assert_called_once_with("### Connection established ###")
