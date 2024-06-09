import React, { useEffect, useRef } from "react";
import "./App.css";
import * as tf from "@tensorflow/tfjs";
import * as posenet from "@tensorflow-models/posenet";
import Webcam from "react-webcam";
import { drawKeypoints, drawSkeleton } from "./utilities";
import axios from 'axios';
import { flipPoseHorizontal } from "@tensorflow-models/posenet/dist/util";

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  
  // load posenet
  const runPosenet = async () => {
    const net = await posenet.load({
      inputResolution: { width: 640, height: 480 },
      scale: 0.5,
    });
    setInterval(() => {
      detect(net);
    }, 2000);
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

      const pose = await net.estimateSinglePose(video, {flipPoseHorizontal: false});
      console.log(pose);

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
    runPosenet();
  }, []);


  return (
    <div className="App">
      <header className="App-header">
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
            width: 640,
            height: 480,
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
            width: 640,
            height: 480,
          }}
        />
      </header>
    </div>
  );
}

export default App;