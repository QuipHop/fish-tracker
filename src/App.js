import React, { useState, useRef } from "react";
import Webcam from "react-webcam";
import * as tf from "@tensorflow/tfjs";

const App = ({ model }) => {
  const [totalFishCount, setTotalFishCount] = useState(0);
  const isRunningRef = useRef(false);
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const detectionIntervalRef = useRef(null);

  const detectFishFromFrame = async () => {
    if (!model || !webcamRef.current || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    // Extract frame from video and draw it on the canvas
    const video = webcamRef.current.video;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Get the frame as a tensor
    const imgTensor = tf.tidy(
      () =>
        tf.browser
          .fromPixels(canvas) // Read from the canvas
          .resizeBilinear([640, 640]) // Resize to model's expected input
          .expandDims(0)
          .toFloat()
          .div(255.0) // Normalize
    );

    try {
      // Perform model prediction
      const [boxes, scores, classes, valid_detections] =
        await model.executeAsync(imgTensor);

      const boxesData = boxes.dataSync();
      const scoresData = scores.dataSync();
      const validDetectionsData = valid_detections.dataSync()[0];

      tf.dispose([boxes, scores, classes, valid_detections]);

      // Prepare predictions for drawing
      const predictions = [];
      for (let i = 0; i < validDetectionsData; ++i) {
        const [x1, y1, x2, y2] = boxesData.slice(i * 4, (i + 1) * 4);

        // Add prediction if confidence > threshold (e.g., 0.5)
        if (scoresData[i] >= 0.2) {
          predictions.push({
            x: x1 * canvas.width,
            y: y1 * canvas.height,
            width: (x2 - x1) * canvas.width,
            height: (y2 - y1) * canvas.height,
            confidence: scoresData[i],
          });
        }
      }

      // Draw predictions on canvas
      drawBoxes(ctx, predictions);

      // Update total fish count
    } catch (error) {
      console.error("Error during prediction:", error);
    } finally {
      tf.dispose(imgTensor);
    }
  };

  const drawBoxes = (ctx, boxes) => {
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height); // Clear the canvas
    boxes.forEach(({ x, y, width, height, confidence }) => {
      ctx.strokeStyle = "limegreen";
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, width, height);

      ctx.font = "14px Arial";
      ctx.fillStyle = "limegreen";
      ctx.fillText(`Confidence: ${(confidence * 100).toFixed(1)}%`, x, y - 5);
    });
  };

  const startDetection = () => {
    if (isRunningRef.current) return;
    isRunningRef.current = true;

    // Process a frame every 500ms
    detectionIntervalRef.current = setInterval(() => {
      if (isRunningRef.current) detectFishFromFrame();
    }, 300);

    console.log("Detection started...");
  };

  const stopDetection = () => {
    isRunningRef.current = false;
    clearInterval(detectionIntervalRef.current);
    detectionIntervalRef.current = null;

    const ctx = canvasRef.current.getContext("2d");
    ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height); // Clear canvas
    console.log("Detection stopped...");
  };

  return (
    <div style={styles.container}>
      <h2>FishAI</h2>
      <div style={styles.videoContainer}>
        <Webcam audio={false} ref={webcamRef} style={styles.webcam} />
        <canvas ref={canvasRef} style={styles.canvas} />
      </div>
      <div style={styles.controls}>
        <button onClick={startDetection} style={styles.button}>
          Start Detection
        </button>
        <button onClick={stopDetection} style={styles.button}>
          Stop Detection
        </button>
      </div>
      <h3>Total Fish Count: {totalFishCount}</h3>
    </div>
  );
};

const styles = {
  container: {
    textAlign: "center",
    fontFamily: "Arial, sans-serif",
  },
  videoContainer: {
    position: "relative",
    display: "inline-block",
  },
  webcam: {
    width: "640px",
    height: "480px",
  },
  canvas: {
    position: "absolute",
    top: 0,
    left: 0,
    width: "640px",
    height: "480px",
  },
  controls: {
    margin: "20px 0",
  },
  button: {
    padding: "10px 20px",
    margin: "0 10px",
    fontSize: "16px",
    cursor: "pointer",
  },
};

export default App;
