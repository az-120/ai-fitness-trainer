from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import logging

app = Flask(__name__)

# logging.basicConfig(level=logging.DEBUG)


CORS(app)


def extract_keypoints(pose, part_name):
    for keypoint in pose:
        if keypoint["part"] == part_name:
            return np.array([keypoint["position"]["x"], keypoint["position"]["y"]])
    return None


def calculate_joint_angles(pose):
    left_hip = extract_keypoints(pose, "leftHip")
    left_knee = extract_keypoints(pose, "leftKnee")
    left_ankle = extract_keypoints(pose, "leftAnkle")

    if left_hip is None or left_knee is None or left_ankle is None:
        return {"error": "Could not find all necessary keypoints"}

    vec_thigh = left_hip - left_knee
    vec_shin = left_ankle - left_knee
    cos_theta = np.dot(vec_thigh, vec_shin) / (
        np.linalg.norm(vec_thigh) * np.linalg.norm(vec_shin)
    )
    angle = np.arccos(cos_theta) * (180 / np.pi)

    return {"left_knee_angle": angle}


@app.route("/analyze", methods=["POST"])
def analyze_pose():
    try:
        data = request.json
        logging.debug(f"Received data: {data}")
        pose = data["pose"]
        angles = calculate_joint_angles(pose)
        logging.debug(f"Calculated angles: {angles}")
        return jsonify(angles)
    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
