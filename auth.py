# auth.py
import streamlit as st
from streamlit_oauth import OAuth2Component
import requests

GOOGLE_CLIENT_ID = "943127606179-0v5o0dd61vt3t3coh8jq68kbsrjvihf6.apps.googleusercontent.com"
GOOGLE_CLIENT_SECRET = "YOUR_CLIENT_SECRET"  # replace this
REDIRECT_URI = "http://localhost:8501"

def google_login():
    oauth2 = OAuth2Component(
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        authorize_endpoint="https://accounts.google.com/o/oauth2/v2/auth",
        token_endpoint="https://oauth2.googleapis.com/token",
    )

    result = oauth2.authorize_button(
        name="login_with_google",
        icon="🔐",
        redirect_uri=REDIRECT_URI,
        scope="openid email profile",
        key="google",
    )

    if result:
        access_token = result.get("token", {}).get("access_token")
        if access_token:
            userinfo_response = requests.get(
                "https://www.googleapis.com/oauth2/v1/userinfo",
                headers={"Authorization": f"Bearer {access_token}"},
            )
            if userinfo_response.ok:
                user_info = userinfo_response.json()
                # Store in session
                st.session_state["user"] = user_info
                st.success(f"✅ Logged in as: {user_info['email']}")
                st.rerun()
