import streamlit as st
import streamlit.components.v1 as components

def voice_input_component(language="hi"):
    st.markdown("### 🎙️ बोलकर सवाल पूछें")

    html_code = f"""
    <div style="text-align: center; padding: 10px;">
        <button id="micBtn" onclick="toggleRecording()" style="
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            font-size: 18px;
            border-radius: 10px;
            cursor: pointer;
        ">🎤 बोलें</button>
        <p id="status" style="margin-top:10px;font-size:16px;color:#666;"></p>
    </div>

    <script>
    let recognition;
    let isRecording = false;

    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {{
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        recognition = new SpeechRecognition();
        recognition.lang = "{language}-IN";
        recognition.interimResults = false;

        recognition.onresult = function(event) {{
            const transcript = event.results[0][0].transcript;

            // Send transcript to Streamlit the CORRECT way
            window.parent.postMessage({{
                type: 'streamlit:setComponentValue',
                value: transcript
            }}, '*');

            // UI feedback
            document.getElementById("micBtn").innerText = "🎤 बोलें";
            document.getElementById("micBtn").style.backgroundColor = "#4CAF50";
            document.getElementById("status").innerText = `✅ आपने कहा: ${{transcript}}`;
            isRecording = false;
        }};
    }}

    function toggleRecording() {{
        if (isRecording) {{
            recognition.stop();
            document.getElementById("micBtn").innerText = "🎤 बोलें";
            document.getElementById("micBtn").style.backgroundColor = "#4CAF50";
            document.getElementById("status").innerText = "⏹️ रिकॉर्डिंग बंद";
            isRecording = false;
        }} else {{
            recognition.start();
            document.getElementById("micBtn").innerText = "🔴 रिकॉर्डिंग जारी...";
            document.getElementById("micBtn").style.backgroundColor = "#DC3545";
            document.getElementById("status").innerText = "🎤 सुन रहा हूं...";
            isRecording = true;
        }}
    }}
    </script>
    """

    transcript = components.html(html_code, height=200)
    return transcript
