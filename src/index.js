import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import '@tensorflow/tfjs-backend-cpu';

import App from './App';
import reportWebVitals from './reportWebVitals';
import '@tensorflow/tfjs-backend-wasm'; 

// Call this function.
window.tflite.setWasmPath(
  'tflite-wasm'
);
// tflite.setWasmPath('/public/tflite-wasm');

// tflite.ready().then(async () => {
//   await tf.setBackend('wasm'); // Set the backend to WASM
//   console.log('WASM backend is ready!');
// });

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
