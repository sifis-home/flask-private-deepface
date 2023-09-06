import os
import platform
from tempfile import NamedTemporaryFile

from flask import Flask, abort, request

from deepface import DeepFace

app = Flask(__name__)


def on_error(ws, error):
    print(error)


def on_close(ws, close_status_code, close_msg):
    print("### Connection closed ###")


def on_open(ws):
    print("### Connection established ###")


@app.route(
    "/check_directory/<file_name>/<privacy_parameter>/<requestor_id>/<requestor_type>/<request_id>",
    methods=["POST"],
)
def check_directory(
    file_name, privacy_parameter, requestor_id, requestor_type, request_id
):
    analyzer_id = platform.node()
    path = request.form.get("path")
    # print(path)
    path = "/app/database"

    if not request.files:
        # If the user didn't submit any files, return a 400 (Bad Request) error.
        abort(400)

    for filename, handle in request.files.items():
        # Create a temporary file.
        # The location of the temporary file is available in `temp.name`.
        temp = NamedTemporaryFile()
        # Write the user's uploaded file to the temporary file.
        # The file will get deleted when it drops out of scope.
        handle.save(temp)

        video_link = temp.name
        # print("video_link:", video_link)

        labels_dict = {}
        # print("listdir:", os.listdir(path))

        labels_dict = DeepFace.private_face_recognition(
            db_path=path,
            source=video_link,
            privacy_parameter=privacy_parameter,
        )

        # print("First:", labels_dict)
        return labels_dict
    else:
        return os.listdir(os.curdir)


@app.route(
    "/cam_face_recognition/<cam_link>/<privacy_parameter>/<requestor_id>/<requestor_type>/<request_id>",
    methods=["POST"],
)
def cam_face_recognition(
    cam_link, privacy_parameter, requestor_id, requestor_type, request_id
):
    analyzer_id = platform.node()
    print(analyzer_id)

    path = request.form.get("path")
    path = "/app/database"
    cam_link = int(cam_link)

    labels_dict = DeepFace.private_face_recognition(
        db_path=path, source=cam_link, privacy_parameter=privacy_parameter
    )

    return labels_dict


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8090)
