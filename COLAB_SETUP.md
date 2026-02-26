# 🚀 Google Colab Setup — Hybrid Text Summarizer

Follow these steps **exactly** to run the AI models on Google Colab's free GPU.

---

## Prerequisites
- A free [ngrok account](https://ngrok.com) — sign up and get your **authtoken** from the [dashboard](https://dashboard.ngrok.com/get-started/your-authtoken).

---

## Step 1: Open Colab

Go to [Google Colab](https://colab.research.google.com/) and create a **New Notebook**.

> ⚠️ **Important**: Click **Runtime → Change runtime type → T4 GPU** to enable the free GPU.

---

## Step 2: Clone Your Repo & Install Dependencies

Paste this into the **first cell** and run it:

```python
# Cell 1: Setup
!git clone https://github.com/aisathish07/Hybrid-Text-Summarization-.git
%cd Hybrid-Text-Summarization-

!pip install -r requirements.txt
!pip install fastapi uvicorn pyngrok nest-asyncio
```

⏳ This will take 2-3 minutes to install everything.

---

## Step 3: Setup ngrok

Paste this into the **second cell**:

```python
# Cell 2: ngrok setup
!ngrok authtoken YOUR_NGROK_TOKEN_HERE
```

> 🔑 Replace `YOUR_NGROK_TOKEN_HERE` with your actual token from https://dashboard.ngrok.com/get-started/your-authtoken

---

## Step 4: Start the API Server

Paste this into the **third cell**:

```python
# Cell 3: Start API server
import nest_asyncio
nest_asyncio.apply()

import sys
sys.path.append('/content/Hybrid-Text-Summarization-')

from pyngrok import ngrok
import uvicorn

# Import the FastAPI app
from api_server import app

# Create ngrok tunnel
public_url = ngrok.connect(8000)
print("=" * 60)
print(f"  🌐 YOUR PUBLIC URL: {public_url}")
print("=" * 60)
print()
print("  👆 Copy this URL and paste it in your Streamlit sidebar!")
print()
print("  Keep this cell running! Don't stop it.")
print("=" * 60)

# Start server (this will block — that's normal!)
uvicorn.run(app, host="0.0.0.0", port=8000)
```

You will see output like:
```
  🌐 YOUR PUBLIC URL: NgrokTunnel: "https://a1b2-34-56-78.ngrok-free.app" -> "http://localhost:8000"
```

---

## Step 5: Connect Your Local Streamlit App

1. Open your **local Streamlit app** at `http://localhost:8501`
2. In the sidebar, find **🖥️ Backend**
3. Select **☁️ Google Colab (Remote)**
4. Paste the ngrok URL (e.g., `https://a1b2-34-56-78.ngrok-free.app`)
5. You should see **✅ Connected to Colab backend!**
6. Paste your text and click **🚀 Generate Summary**

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ngrok authtoken` error | Make sure you signed up at ngrok.com and used the correct token |
| Connection refused | Make sure Cell 3 is still running in Colab (don't stop it) |
| Timeout errors | Colab may be slow on first run as it downloads models (~5min) |
| `ModuleNotFoundError` | Re-run Cell 1 to reinstall dependencies |
| Colab disconnects | Colab has a ~90 min idle timeout. Just re-run all cells if it disconnects |

---

## Notes
- 🆓 Colab's free T4 GPU makes inference ~10x faster than CPU
- ⏰ Free Colab sessions last ~12 hours max, ~90 min idle timeout
- 💾 Models are re-downloaded each time Colab resets (cached within the session)
- 🔒 Your text is sent to the Colab server over HTTPS (ngrok tunnel)
