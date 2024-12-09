// src/WebcamComponent.js
import React from 'react';
import Webcam from 'react-webcam';

const WebcamComponent = React.forwardRef((props, ref) => (
    <Webcam
        audio={false}
        ref={ref}
        screenshotFormat="image/jpeg"
        videoConstraints={{
            width: 640,
            height: 480,
            facingMode: "user",
        }}
        style={{ border: '1px solid black' }}
    />
));

export default WebcamComponent;
