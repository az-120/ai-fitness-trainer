from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)

CORS(app)

CONFIDENCE_THRESHOLD = 0.6


def extract_keypoints(pose, part_name):
    for keypoint in pose:
        if keypoint["part"] == part_name:
            if keypoint["score"] < CONFIDENCE_THRESHOLD:
                return None
            return np.array([keypoint["position"]["x"], keypoint["position"]["y"]])
    return None


def calculate_joint_angles(pose):
    hip = extract_keypoints(pose, "leftHip")
    knee = extract_keypoints(pose, "leftKnee")
    ankle = extract_keypoints(pose, "leftAnkle")
    shoulder = extract_keypoints(pose, "leftShoulder")

    if hip is None or knee is None or ankle is None or shoulder is None:
        return {"error": "Could not find all necessary keypoints"}

    vec_thigh = hip - knee
    vec_shin = ankle - knee
    cos_theta_knee = np.dot(vec_thigh, vec_shin) / (
        np.linalg.norm(vec_thigh) * np.linalg.norm(vec_shin)
    )
    left_knee_angle = np.arccos(cos_theta_knee) * (180 / np.pi)

    vec_torso = shoulder - hip
    cos_theta_hip = np.dot(vec_thigh, vec_torso) / (
        np.linalg.norm(vec_thigh) * np.linalg.norm(vec_torso)
    )
    left_hip_angle = np.arccos(cos_theta_hip) * (180 / np.pi)

    return {"left_knee_angle": left_knee_angle, "left_hip_angle": left_hip_angle}


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
