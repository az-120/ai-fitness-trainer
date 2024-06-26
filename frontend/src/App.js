import React, { useState, useEffect, useRef } from "react";
import "./App.css";
import * as tf from "@tensorflow/tfjs";
import * as posenet from "@tensorflow-models/posenet";
import { drawKeypoints, drawSkeleton } from "./utilities";
import { flipPoseHorizontal } from "@tensorflow-models/posenet/dist/util";
import axios from 'axios';
import Webcam from "react-webcam";

function App() {
  const [selectedExercise, setSelectedExercise] = useState(null);
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);

  const handleExerciseSelect = (exercise) => {
    setSelectedExercise(exercise);
  };
  
  // load posenet
  const runPosenet = async () => {
    const net = await posenet.load({
      inputResolution: { width: 1280, height: 720 },
      scale: 0.5,
    }
  );
    setInterval(() => {
      detect(net);
    }, 2000); // lower this later
  };

  const detect = async (net) => {
    if (
      typeof webcamRef.current !== "undefined" &&
      webcamRef.current !== null &&
      webcamRef.current.video.readyState === 4
    ) {
      const video = webcamRef.current.video;
      const videoWidth = webcamRef.current.video.videoWidth;
      const videoHeight = webcamRef.current.video.videoHeight;

      video.width = videoWidth
      video.height = videoHeight

      const pose = await net.estimateSinglePose(video, {flipPoseHorizontal: true});
      // console.log(pose);

      drawCanvas(pose, video, videoWidth, videoHeight, canvasRef);

      try {
        // send data to flask server
        const response = await axios.post('http://localhost:5000/analyze', { pose: pose.keypoints });
        const angles = response.data;

        // feedback from backend calcs
        giveFeedback(angles);
      } catch (error) {
        console.error("There was an error sending the data!", error);
      }
    }
  };

  const drawCanvas = (pose, video, videoWidth, videoHeight, canvas) => {
    const ctx = canvas.current.getContext("2d");
    canvas.current.width = videoWidth;
    canvas.current.height = videoHeight;

    drawKeypoints(pose["keypoints"], 0.6, ctx);
    drawSkeleton(pose["keypoints"], 0.7, ctx);
  };

  const giveFeedback = (angles) => {
    // TO DO: feedback
    console.log('Feedback:', angles);
  };

  useEffect(() => {
    if (selectedExercise) {
      runPosenet();
    }
  }, [selectedExercise]);


  return (
    <div className="App">
      <header className="App-header">
        {!selectedExercise ? (
          <button className="exercise-button" onClick={() => handleExerciseSelect('squat')}>Select Squat</button>
        ) : (
          <>
            <Webcam ref={webcamRef}
              screenshotFormat="image/png"
              audio={false}
              style={{
                position: "absolute",
                marginLeft: "auto",
                marginRight: "auto",
                left: 0,
                right: 0,
                textAlign: "center",
                zindex: 9,
                width: 1280,
                height: 720,
              }}
            />
            <canvas
              ref={canvasRef}
              style={{
                position: "absolute",
                marginLeft: "auto",
                marginRight: "auto",
                left: 0,
                right: 0,
                textAlign: "center",
                zindex: 9,
                width: 1280,
                height: 720,
              }}
            />
          </>
        )}
      </header>
    </div>
  );
}

export default App;