from auth import google_login
import streamlit as st

if "user" not in st.session_state:
    st.markdown("""
    <style>
        body {
            background: linear-gradient(to top right, #d1ffea, #f4fff9);
        }
        .login-box {
            max-width: 450px;
            margin: 80px auto;
            background: white;
            padding: 30px 40px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            text-align: center;
            font-family: 'Segoe UI', sans-serif;
        }
        .login-box h2 {
            color: #2e8b57;
            font-size: 30px;
            font-weight: 700;
        }
        .login-box p {
            font-size: 16px;
            color: #555;
        }
        .google-btn {
            margin-top: 20px;
        }
        .brand-logo {
            width: 80px;
            margin-bottom: 10px;
        }
    </style>

    <div class="login-box">
        <img src="https://cdn-icons-png.flaticon.com/128/756/756669.png" class="brand-logo" alt="KRISH Logo"/>
        <h2>🌾 KRISH कृषि सहायक</h2>
        <p>स्मार्ट खेती के लिए डिजिटल साथी!<br>
        मंडी भाव, फसल सुझाव, मौसम और बहुत कुछ...</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
            google_login()  # Renders the Google sign-in button
    st.markdown("</div>", unsafe_allow_html=True)

    st.stop()

