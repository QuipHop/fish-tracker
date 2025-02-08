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

    const video = webcamRef.current.video;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    // Set canvas dimensions to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Get the image tensor
    const imgTensor = tf.tidy(
      () =>
        tf.browser
          .fromPixels(video) // Directly from video
          .resizeBilinear([640, 640])
          .expandDims(0)
          .toFloat()
          .div(255.0) // Normalize
    );

    try {
      const prediction = await model.predict(imgTensor);

      // Process YOLO predictions
      const reshaped = prediction.reshape([5, -1]);
      const boxes = reshaped.slice([0, 0], [4, -1]).transpose();
      const scores = reshaped.slice([4, 0], [1, -1]).reshape([-1]);

      const nmsIndices = await tf.image.nonMaxSuppressionAsync(
        boxes,
        scores,
        10, // Keep top 10
        0.7, // IoU threshold
        0.5 // Score threshold
      );

      // Extract boxes and scores
      const selectedBoxes = tf.gather(boxes, nmsIndices);
      const selectedScores = tf.gather(scores, nmsIndices);

      const boxesArray = await selectedBoxes.array();
      const scoresArray = await selectedScores.array();

      console.log("boxesArray ", boxesArray);
      // Map predictions
      const predictions = boxesArray.map((box, i) => ({
        x: box[0] * canvas.width,
        y: box[1] * canvas.height,
        width: (box[2] - box[0]) * canvas.width,
        height: (box[3] - box[1]) * canvas.height,
        confidence: scoresArray[i],
      }));
      console.log("predictions ", predictions);
      // Draw detections
      drawBoxes(ctx, predictions);
      setTotalFishCount((prevCount) => prevCount + predictions.length);
    } catch (error) {
      console.error("Error during prediction:", error);
    } finally {
      tf.dispose([imgTensor]); // Dispose tensors
    }
  };

  const drawBoxes = (ctx, boxes) => {
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height); // Clear only necessary
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

    detectionIntervalRef.current = setInterval(() => {
      if (isRunningRef.current) detectFishFromFrame();
    }, 100); // Run detection every 500ms
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
  canvas: {
    position: "absolute",
    top: 0,
    left: 0,
    width: "100%",
    height: "100%",
    zIndex: 2,
    objectFit: "contain", // Ensures correct scaling
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
