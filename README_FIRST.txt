
CARE FLOW AI — FINAL STRUCTURED PORTAL

WHAT THIS VERSION DOES
- Shows a welcoming landing page before dashboard access
- Supports Google sign-in once deployed and OAuth is configured
- Keeps the dashboard unavailable until sign-in
- Includes local Preview Mode only for your own testing before deployment
- Organizes the system into:
  1. Dashboard
  2. Predict Waiting Time
  3. Model Performance
  4. About the Project
  5. My Profile
- Includes a professional hospital/research presentation style

LOCAL TESTING
1. Extract the folder.
2. Open Command Prompt in this folder.
3. Install dependencies:
   python -m pip install --no-cache-dir --default-timeout=300 -r requirements.txt
4. Run:
   python -m streamlit run app.py
5. Open:
   http://localhost:8501

PUBLIC ACCESS WITH GOOGLE SIGN-IN
To give other people a link and require Google/Gmail sign-in, deploy the app to a public HTTPS host.
Then configure Google OAuth credentials and add them to Streamlit secrets.

Never place a real client_secret in a public GitHub repository or shared ZIP.

RESEARCH NOTE
The current trained model uses the generated development dataset prepared for prototype testing.
Do not describe it as validated on real hospital patient records.
