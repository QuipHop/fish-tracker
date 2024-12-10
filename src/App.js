import React, { useState, useRef } from "react";
import Webcam from "react-webcam";
import * as tf from "@tensorflow/tfjs";

const App = ({ model }) => {
  const [totalFishCount, setTotalFishCount] = useState(0);
  const isRunningRef = useRef(false); // Use a ref to manage the running state
  const animationFrameIdRef = useRef(null); // Store the animation frame ID
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);

  const detectFish = async () => {
    if (!model || !webcamRef.current || !canvasRef.current) return;

    const video = webcamRef.current.video;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const imgTensor = tf.browser
      .fromPixels(video)
      .resizeBilinear([320, 320])
      .expandDims(0)
      .toFloat()
      .div(255.0);

    try {
      const outputs = await model.predict(imgTensor);

      const numDetections = outputs["StatefulPartitionedCall:0"].dataSync()[0];
      const scores = outputs["StatefulPartitionedCall:1"].dataSync();
      const boxes = outputs["StatefulPartitionedCall:3"].dataSync();
      const classes = outputs["StatefulPartitionedCall:2"].dataSync();

      const confidenceThreshold = 0.5;
      const labelMap = ["fish"];
      const detections = [];

      for (let i = 0; i < numDetections; i++) {
        if (scores[i] > confidenceThreshold) {
          const ymin = boxes[i * 4 + 0] * video.videoHeight;
          const xmin = boxes[i * 4 + 1] * video.videoWidth;
          const ymax = boxes[i * 4 + 2] * video.videoHeight;
          const xmax = boxes[i * 4 + 3] * video.videoWidth;

          detections.push({
            class: labelMap[classes[i]],
            confidence: scores[i],
            bbox: { ymin, xmin, ymax, xmax },
          });

          // Draw bounding boxes
          ctx.strokeStyle = "#00FF00";
          ctx.lineWidth = 2;
          ctx.strokeRect(xmin, ymin, xmax - xmin, ymax - ymin);
          ctx.fillStyle = "#00FF00";
          ctx.font = "14px Arial";
          ctx.fillText(
            `${labelMap[classes[i]]} (${(scores[i] * 100).toFixed(1)}%)`,
            xmin,
            ymin > 10 ? ymin - 5 : ymin + 15
          );
        }
      }

      // Update fish count efficiently
      setTotalFishCount(detections.length);

      // Dispose of tensors
      Object.values(outputs).forEach((tensor) => tf.dispose(tensor));
    } catch (error) {
      console.error("Error during prediction:", error);
    } finally {
      tf.dispose(imgTensor);
    }
  };

  const startDetection = () => {
    if (isRunningRef.current) return; // Prevent multiple loops
    isRunningRef.current = true;

    const runDetection = async () => {
      if (!isRunningRef.current) return; // Stop the loop if not running
      await detectFish();
      animationFrameIdRef.current = requestAnimationFrame(runDetection); // Continue the loop
    };

    animationFrameIdRef.current = requestAnimationFrame(runDetection); // Start the loop
    console.log("Detection started...");
  };

  const stopDetection = () => {
    if (!isRunningRef.current) return; // Prevent unnecessary calls
    isRunningRef.current = false; // Mark detection as stopped

    if (animationFrameIdRef.current) {
      cancelAnimationFrame(animationFrameIdRef.current); // Stop the loop
      animationFrameIdRef.current = null; // Clear frame reference
    }

    // Clear the canvas
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height); // Clear drawings
    }

    console.log("Detection stopped...");
  };

  return (
    <div style={styles.container}>
      <div style={styles.videoContainer}>
        <Webcam audio={false} ref={webcamRef} style={styles.webcam} />
        <canvas ref={canvasRef} style={styles.canvas} />
      </div>
      <div style={styles.infoContainer}>
        <h3>Total Fish Count: {totalFishCount}</h3>
      </div>
      <div style={styles.buttonContainer}>
        <button onClick={startDetection} style={styles.button}>
          Start Detection
        </button>
        <button onClick={stopDetection} style={styles.button}>
          Stop Detection
        </button>
      </div>
    </div>
  );
};

const styles = {
  container: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    width: "100vw",
    height: "100vh",
    backgroundColor: "#f5f5f5",
  },
  videoContainer: {
    position: "relative",
    width: "640px",
    height: "480px",
  },
  webcam: {
    position: "absolute",
    top: 0,
    left: 0,
    width: "100%",
    height: "100%",
    zIndex: 1,
  },
  canvas: {
    position: "absolute",
    top: 0,
    left: 0,
    width: "100%",
    height: "100%",
    zIndex: 2,
    pointerEvents: "none",
  },
  infoContainer: {
    marginTop: "10px",
    fontSize: "18px",
    color: "#333",
  },
  buttonContainer: {
    marginTop: "20px",
    display: "flex",
    gap: "20px",
  },
  button: {
    padding: "10px 20px",
    fontSize: "16px",
    backgroundColor: "#007bff",
    color: "#fff",
    border: "none",
    borderRadius: "5px",
    cursor: "pointer",
  },
};

export default App;
