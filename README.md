## CLI AI Conversational Agent (ElevenLabs)

Terminal-native voice and text chat that connects to an ElevenLabs Agent over WebSocket. It streams your microphone audio, plays the agent's audio responses in real-time, and renders live transcripts/status in a clean CLI using Rich.

Built for fast, reliable operator workflows in the terminal.

### Highlights
- Real-time mic capture and audio playback (16 kHz PCM)
- Type-to-chat alongside voice input
- Live status: connection, speaking/listening, VAD score, events, errors
- Works with signed URLs from your backend or with a public `AGENT_ID`
- Multiple languages supported by the agent (see list below)

### Requirements
- Python 3.11+
- PortAudio (installed by `sounddevice` on macOS; on Linux install system package `portaudio` if needed)

### Files
- `main.py`: CLI agent
- `server.py`: FastAPI backend to fetch ElevenLabs signed URL
- `requirements.txt`: Python dependencies
- `app.js`: Example web app (not required for CLI)

### Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set environment and run the backend (for signed URLs):
```bash
export XI_API_KEY=YOUR_11LABS_API_KEY
export AGENT_ID=YOUR_AGENT_ID
export BACKEND_URL=http://localhost:8000
uvicorn server:app --reload
```

### Run the CLI
```bash
# Using signed URL via backend
python main.py

# Or connect directly with a public Agent ID (no backend):
python main.py --agent-id "$AGENT_ID"

# List audio devices and then choose indices
python -c "import sounddevice as sd; print(sd.query_devices())"
python main.py --input-device 1 --output-device 0

# Optional tweaks
python main.py --sample-rate 48000    # if your device prefers 48k
python main.py --no-mic               # type-only chat
```

Controls: Press Enter to send typed messages. Ctrl+C to quit.

### CLI Options
- `--backend-url`: Backend providing `/api/signed-url` (default `http://localhost:8000` or `BACKEND_URL` env)
- `--agent-id`: Public Agent ID (skips signed URL flow; uses ElevenLabs WSS directly)
- `--sample-rate`: Audio sample rate (default 16000)
- `--input-device` / `--output-device`: `sounddevice` indices
- `--no-mic`: Disable microphone streaming

### Troubleshooting
- If you see an audio error like "Invalid number of channels":
  - Choose a different `--output-device` index
  - Try `--sample-rate 48000`
  - The CLI auto-detects mono/stereo and will duplicate mono into stereo if needed

### Tools via Zapier MCP (optional)
If connected to a Zapier MCP server and authorized, the agent can invoke these tools:

- Google Docs: create, update, share documents
- Google Sheets: create sheets, update cells, run lookups
- Google Calendar: find/add/update events, busy times
- Google Meet: schedule meetings and share links
- Google Drive: search/upload/manage files
- Google Forms: create forms
- Gmail: draft/reply/send, label, find emails
- Telegram: send messages/photos/polls
- WhatsApp: send messages (where available)

Note: Tool use depends on your MCP setup, credentials, and agent configuration.

### Language support
The ElevenLabs Agent can converse in multiple languages; the CLI streams/plays audio agnostic of language:

- Dutch, Finnish, Turkish, Russian, Tamil, Croatian, Romanian, Korean, Norwegian, Hungarian, Chinese, Japanese, Filipino, Ukrainian, Italian, Swedish, Indonesian, Arabic, Portuguese (Brazil), Spanish, Hindi, Greek, Bulgarian, Danish, Malay, Vietnamese, Portuguese, German, Czech, Slovak, French, Polish

### Privacy
When enabled, your microphone audio is streamed to ElevenLabs for real-time processing. Do not run this in sensitive environments unless you have permission.

### References
- Quickstart: [ElevenLabs Agents Quickstart](https://elevenlabs.io/docs/agents-platform/quickstart)
- WebSocket API: [Agents Platform WebSocket](https://elevenlabs.io/docs/agents-platform/api-reference/agents-platform/websocket)


