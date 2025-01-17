"use client";
import { useEffect, useState } from "react";
import axios from "axios";

export default function Home() {
  const [threatMessage, setThreatMessage] = useState("Waiting for threat assessment...");
  const [videoStream, setVideoStream] = useState(null);

  useEffect(() => {
    // Access the user's camera (not directly used in the frontend as backend streams it)
    const videoElement = document.getElementById("videoElement");
    videoElement.src = "http://127.0.0.1:8000/video_feed"; // Backend stream for video
    
    // Polling the backend every 5 seconds for threat level updates
    const intervalId = setInterval(() => {
      axios
        .get("http://127.0.0.1:8000/compute-threat")
        .then((response) => {
          const result = response.data.message;
          setThreatMessage(result);
        })
        .catch((error) => {
          console.error("Error fetching threat data:", error);
        });
    }, 5000); // Poll every 5 seconds

    // Clean up when the component is unmounted
    return () => clearInterval(intervalId);
  }, []);

  return (
    <div style={{ textAlign: "center" }}>
      <h1>MacAi LipReading AI</h1>
      <div>
        <video
          id="videoElement"
          autoPlay
          width="80%"
          style={{ maxWidth: "640px" }}
        ></video>
      </div>
      <div
        id="threatMessage"
        style={{
          marginTop: "20px",
          fontSize: "20px",
          fontWeight: "bold",
          color: threatMessage.includes("Threat") ? "red" : "green",
        }}
      >
        {threatMessage}
      </div>
    </div>
  );
}
