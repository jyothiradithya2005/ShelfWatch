# ShelfWatch

ShelfWatch counts groceries from a photo, a video, or a live camera, and sends you a text message when something runs low. It can also count people instead.

It combines two detection models:

- **`best.pt`**: a custom model trained on over 100 grocery items, such as tomato, onion, carrot, milk and eggs.
- **`yolo11n.pt`**: a general model trained on the COCO dataset. It adds about 70 everyday objects (bottles, cups, bowls, phones and so on) and is also used to detect people.

## Requirements

- Python 3.10 or newer
- A webcam or an RTSP/HTTP camera stream, for live detection
- A Twilio account, for SMS alerts (optional)

## Setup

1. Open a terminal in the project folder:

   ```bash
   cd realtime_stock
   ```

2. Create and activate a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

   On Windows, activate it with `venv\Scripts\activate` instead.

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Check that `best.pt` is in the project folder, next to `app.py`. You don't need to download `yolo11n.pt` yourself. If it's missing, it downloads automatically the first time the app runs.

5. (Optional) Set up SMS alerts. See [SMS alerts](#sms-alerts) below.

## Run the app

With the virtual environment active, run:

```bash
streamlit run app.py
```

The app opens in your browser at http://localhost:8501. To stop it, press **Ctrl+C** in the terminal.

If you edit any `.py` file or `.streamlit/secrets.toml`, restart the app with **Ctrl+C** and `streamlit run app.py` again. Refreshing the browser page isn't always enough.

## Using the app

### Sidebar settings

| Setting | What it does |
| --- | --- |
| **Detect: Objects / People** | **Objects** counts groceries and everyday items and ignores people. **People** counts only people. |
| **Include everyday objects** | Groceries are always detected. Turn this on to also detect everyday objects like bottles, cups and bowls. Only shown in Objects mode. |
| **Confidence** | How sure the model must be before it counts something, from 0.10 to 0.90. Lower it if items are being missed; raise it if things are counted that aren't there. The default is 0.35. |
| **Low stock at or below** | An item with this many or fewer is marked **Low**. An item with none is marked **Out of stock**. |
| **Items to track** | The items shown in the stock table and checked for SMS alerts. |
| **Send SMS alerts** | Sends a text when a tracked item runs low. Off by default. |
| **Reset alerts** | Lets items that have already sent an alert send one again. |

### Tabs

- **Photo**: upload a photo or take one with your camera. The app shows the detected items, the stock levels and a count of each item.
- **Video**: upload a video (MP4, MOV, AVI or MKV) and click **Analyse video**. Use **Analyse every Nth frame** to trade accuracy for speed. When it finishes, you can download the video with boxes drawn on it.
- **Live camera**: enter `0` for your computer's webcam, or paste an RTSP or HTTP stream URL, then turn on **Start live detection**.
- **Alert log**: every SMS alert the app has sent or tried to send, and whether it worked.

### Live and cumulative counts

The Video and Live camera tabs show two tables:

- **Live count**: what's in view right now. It's smoothed over the last 15 frames so the numbers don't flicker.
- **Cumulative count**: how many different objects have been seen since the run started. Each object is tracked across frames, so it's only counted once, however long it stays in view. An object has to appear in 3 frames in a row to be counted. The count restarts each time you start live detection or analyse a new video.

The stock table and SMS alerts use the live count.

## SMS alerts

Alerts are sent with [Twilio](https://www.twilio.com/). Add your details to `.streamlit/secrets.toml`:

```toml
[twilio]
sid = "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
auth_token = "your_auth_token"
messaging_service_sid = "MGxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
to_number = "+91xxxxxxxxxx"
```

- Find `sid` and `auth_token` on your Twilio console dashboard.
- To send from a Twilio phone number instead of a Messaging Service, replace `messaging_service_sid` with `from_number = "+1xxxxxxxxxx"`.
- `to_number` is the phone that receives the alerts, including the country code.

Restart the app after editing this file. The sidebar should say **Twilio is configured**. If it's not configured, alerts still appear in the Alert log but no text is sent.

Each tracked item sends one text when it drops to or below the low-stock number. It can alert again after it goes back above that number, or after you click **Reset alerts**. Each text includes a reorder link.

On a Twilio trial account, you can only text phone numbers you've verified in the Twilio console.

> Keep `secrets.toml` private. Don't commit it to git or share it. If your auth token gets exposed, create a new one in the Twilio console.

## Project files

| File | Purpose |
| --- | --- |
| `app.py` | The Streamlit app: sidebar, tabs and tables |
| `detector.py` | Runs both models, merges their results, and counts and tracks objects |
| `alerts.py` | Decides stock status and sends SMS alerts through Twilio |
| `config.py` | Model files, default settings and reorder links |
| `best.pt` | Custom grocery detection model |
| `yolo11n.pt` | COCO model for everyday objects and people (downloaded automatically) |
| `.streamlit/config.toml` | App colour theme |
| `.streamlit/secrets.toml` | Twilio credentials |

## Customising

These settings are in `config.py`:

- `DEFAULT_CONFIDENCE`, `DEFAULT_THRESHOLD`: the starting values for the sidebar sliders
- `DEFAULT_TRACKED`: the items tracked when the app opens
- `SMOOTHING_WINDOW`: how many frames the live count is smoothed over
- `REORDER_LINKS`: the shop link sent for each item. Items without a link use a BigBasket search.
- `COCO_MODEL`: the COCO model file. `yolo11s.pt` or `yolo11m.pt` are more accurate but slower.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `ImportError` after editing code | Restart the app with **Ctrl+C** and `streamlit run app.py`. |
| "No model file at …" | Put `best.pt` in the project folder next to `app.py`. |
| "Couldn't open camera 0" | Close other apps using the webcam. On macOS, give your terminal camera access in **System Settings → Privacy & Security → Camera**. |
| Items are missed | Lower **Confidence**, or improve the lighting. |
| Wrong items are counted | Raise **Confidence**. |
| Live camera is slow | Turn off **Include everyday objects** so only one model runs. For videos, raise **Analyse every Nth frame**. |
| "Twilio isn't configured" | Check that every field in `.streamlit/secrets.toml` is filled in, then restart the app. |
| Alert log says "Failed: …" | Check your Twilio credentials, your balance and, on a trial account, that the number is verified. |
