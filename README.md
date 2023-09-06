# WP4 Analytic: Privacy-Aware Face Recognition

[![Actions Status][actions badge]][actions]
[![CodeCov][codecov badge]][codecov]
[![LICENSE][license badge]][license]

<!-- Links -->
[actions]: https://github.com/sifis-home/flask_private_deepface/actions
[codecov]: https://codecov.io/gh/sifis-home/flask_private_deepface
[license]: LICENSES/MIT.txt

<!-- Badges -->
[actions badge]: https://github.com/sifis-home/flask_private_deepface/workflows/flask_private_deepface/badge.svg
[codecov badge]: https://codecov.io/gh/sifis-home/flask_private_deepface/branch/master/graph/badge.svg
[license badge]: https://img.shields.io/badge/license-MIT-blue.svg

Accurate person recognition is essential for various purposes, including identifying home residents, guests, and potential intruders. It enables the provision of personalized smart services based on individual identities. Typically, face recognition models are trained using images of the home's resident users. When a person enters the home or a specific room, their face image is detected and compared against the trained model, allowing for classification based on extracted facial features. This process ensures reliable and efficient person recognition in a home environment. 
The face recognition algorithm utilizes images and video frames captured by surveillance cameras within a controlled environment. This includes both the camera of the controlled device itself and other surveillance cameras deployed in the surroundings. In addition, recorded videos and captured images can also be used as input data. The algorithm requires a database directory that contains the identities of authorized users. By processing this input data, the face recognition system can accurately identify and verify individuals within the controlled environment. 
The used face recognition model is [DeepFace](https://github.com/serengil/deepface), an open-source model which uses a deep learning mechanism for face recognition. The core of the architecture is a Convolutional Neural Network (CNN) model that takes image data as input and produces the relative representation as output and verifies it with the representations of authorized identities. The model has been trained on a large dataset for face recognition based on a distance metric between face representations. This model has been validated with a set of experiments on a well-known dataset, the Labelled Faces in the Wild (LFW) dataset. The workflow starts with image capture and anonymization using Gaussian blurring on the user side. The resulting anonymized images are processed for face recognition starting with a face detector OpenCV to detect all faces within an image. Detected faces are aligned and then converted to vectors. Finally, the faces are verified by comparing their representations with the representations of face images stored in the database. We use VGG-Face deep-learning model for face recognition.

The SIFIS-Home face recognition analytic is composed of a pipeline of the following four components:  
- **Face Detection**.
- **Face Alignment**.
- **Face Representation**.
- **Face Verification**.
 
## Deploying

### Privacy-Aware Face Recognition in a container

Privacy-Aware Face Recognition is intended to run in a docker container on port 8090. The Dockerfile at the root of this repo describes the container. To build and run it execute the following commands:

`docker build -t flask_private_deepface .`

`docker-compose up`

## REST API of Privacy-Aware Face Recognition

Description of the REST endpoint available while Privacy-Aware Face Recognition is running.

---

#### GET /check_directory

Description: Returns the result whether the face in an image or a video frame is identified or not and the identity.

Command:

`curl -F "path=/app/database" -F "file=@file_location" http://localhost:8090/check_directory/<file_name.mp4>/<privacy_parameter>/<requestor_id>/<requestor_type>/<request_id>`

Sample:

`curl -F "path=/app/database" -F "file=@file_location" http://localhost:8090/check_directory/sample.mp4/17/33466553786f48cb72faad7b2fb9d0952c97/NSSD/2023061906001633466553786f48cb72faad7b2fb9d0952c97`

#### GET /cam_face_recognition

Description: Returns the result whether the face in an CAM Link is identified or not and the identity.

Command:

`curl -F "path=/app/database" http://localhost:8090/cam_face_recognition/<cam_link>/<privacy_parameter>/<requestor_id>/<requestor_type>/<request_id>`

Sample:

`curl -F "path=/app/database" http://localhost:8090/cam_face_recognition/0/17/33466553786f48cb72faad7b2fb9d0952c97/NSSD/2023061906001633466553786f48cb72faad7b2fb9d0952c97`

---
## License

Released under the [MIT License](LICENSE).

## Acknowledgements

This software has been developed in the scope of the H2020 project SIFIS-Home with GA n. 952652.