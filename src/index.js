import React from "react";
import ReactDOM from "react-dom/client";
import "./index.css";
import "@tensorflow/tfjs-backend-cpu";
import "@tensorflow/tfjs-backend-wasm";
import * as tf from "@tensorflow/tfjs";
import App from "./App";
import reportWebVitals from "./reportWebVitals";

const initializeTF = async () => {
  window.tflite.setWasmPath("/tflite-wasm/");
  await tf.setBackend("wasm"); // Set the backend to WASM
  await tf.ready();
  console.log("TensorFlow.js WASM backend is ready!");

  const model = await window.tflite.loadTFLiteModel(
    "/models/fish_detection/detect.tflite"
  );
  console.log("Model loaded successfully!");
  return model;
};

initializeTF().then((model) => {
  const root = ReactDOM.createRoot(document.getElementById("root"));
  root.render(
    <React.StrictMode>
      <App model={model} />
    </React.StrictMode>
  );
});

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
