import os
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv

from google import genai
from google.genai import types
from google.cloud import aiplatform

load_dotenv()

# Load credentials and setup
PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")
ENDPOINT = os.getenv("MODEL_ENDPOINT")  # Full endpoint ID from Vertex AI

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def form_get():
    return """
    <html>
        <body>
            <h2>Translate with Gemini</h2>
            <form method="post">
                <textarea name="text" rows="4" cols="50" placeholder="Enter text to translate..."></textarea><br>
                <input type="submit" value="Translate">
            </form>
        </body>
    </html>
    """

@app.post("/", response_class=HTMLResponse)
async def translate_post(text: str = Form(...)):
    # Set up the Vertex AI client
    client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=LOCATION
    )

    contents = [
        types.Content(
            role="user",
            parts=[types.Part(text=text)]
        )
    ]

    config = types.GenerateContentConfig(
        temperature=0.5,
        top_p=0.8,
        max_output_tokens=1024,
        safety_settings=[
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_LOW_AND_ABOVE"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_LOW_AND_ABOVE")
        ]
    )

    # Call the fine-tuned model
    try:
        response = client.models.generate_content(
            model=ENDPOINT,
            contents=contents,
            config=config
        )
        translated_text = response.text
    except Exception as e:
        translated_text = f"Error: {str(e)}"

    return f"""
    <html>
        <body>
            <h2>Translation Result</h2>
            <p><strong>Input:</strong> {text}</p>
            <p><strong>Translation:</strong> {translated_text}</p>
            <a href="/">Translate Another</a>
        </body>
    </html>
    """
