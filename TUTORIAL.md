## Tutorial: CLI AI Conversational Agent (ElevenLabs)

This tutorial walks you through installing, configuring, and running the terminal-native conversational agent that connects to an ElevenLabs Agent over WebSocket. You can speak via your microphone or type, and hear the agent's responses in real time, with transcripts and statuses rendered directly in your terminal.

### What you'll build
- A CLI that streams mic audio to ElevenLabs and plays back agent audio
- Live transcripts: user speech, tentative and final agent responses
- Rich terminal UI with connection status, speaking/listening mode, VAD score
- Optional direct connection to a public Agent ID (no backend needed)

### Prerequisites
- Python 3.11+
- macOS/Linux with microphone and speakers (or a virtual device)
- PortAudio (installed automatically by `sounddevice` on macOS; install system package on Linux if necessary)
- ElevenLabs account, API key, and an Agent

If you don't have an Agent yet, create one and configure voice and behavior in the ElevenLabs dashboard. See the official quickstart linked at the end.

### 1) Clone and install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Configure ElevenLabs
You'll need:
- `XI_API_KEY`: Your ElevenLabs API key
- `AGENT_ID`: The Agent you want to connect to

You can store them in a `.env` file or export them in your shell:
```bash
export XI_API_KEY=YOUR_11LABS_API_KEY
export AGENT_ID=YOUR_AGENT_ID
```

### 3) Start the backend for signed URLs (recommended)
The backend provides a signed URL endpoint required by private agents and simplifies auth.

```bash
export BACKEND_URL=http://localhost:8000
uvicorn server:app --reload
```

Endpoints:
- `GET /api/signed-url` → `{ "signedUrl": "wss://..." }`
- `GET /api/getAgentId` → `{ "agentId": "..." }`

### 4) Run the CLI
Using signed URL via backend (private or public agents):
```bash
python main.py
```

Direct connection using a public Agent ID (skips backend):
```bash
python main.py --agent-id "$AGENT_ID"
```

Useful options:
- `--backend-url` Override backend URL (default `BACKEND_URL` env or `http://localhost:8000`)
- `--sample-rate` Audio sample rate (default `16000`; try `48000` if your device prefers 48 kHz)
- `--input-device` / `--output-device` Sound device indices (see section below)
- `--no-mic` Disable mic streaming and use type-only chat

### 5) Pick audio devices
List your system devices and note the indices:
```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
```
Run the CLI specifying indices:
```bash
python main.py --input-device 1 --output-device 0
```

The agent plays audio in mono or stereo depending on your output device; mono is duplicated to stereo automatically when needed.

### 6) Use the CLI
- After connecting, speak to the agent; you’ll see live transcripts and hear the reply.
- Press Enter to send a typed message at any time.
- Press Ctrl+C to disconnect and exit.

UI indicators show connection status, speaking/listening mode, VAD score, events processed, and any errors.

### 7) Multilingual conversations
The agent can converse in many languages; the CLI is language-agnostic and streams audio/text as-is. Supported languages include: Dutch, Finnish, Turkish, Russian, Tamil, Croatian, Romanian, Korean, Norwegian, Hungarian, Chinese, Japanese, Filipino, Ukrainian, Italian, Swedish, Indonesian, Arabic, Portuguese (Brazil), Spanish, Hindi, Greek, Bulgarian, Danish, Malay, Vietnamese, Portuguese, German, Czech, Slovak, French, Polish.

### 8) Tools via Zapier MCP (optional)
If your agent is configured to use a Zapier MCP server and the proper credentials are provided, it can call tools to act in your apps:

- Google Docs: create, update, share documents
- Google Sheets: create sheets, update cells, run lookups
- Google Calendar: find/add/update events, busy times
- Google Meet: schedule meetings and share links
- Google Drive: search/upload/manage files
- Google Forms: create forms
- Gmail: draft/reply/send, label, find emails
- Telegram: send messages/photos/polls
- WhatsApp: send messages (where available)

Setup depends on your MCP configuration and access—refer to your MCP/Zapier integration docs and ensure the agent has permission to call those tools.

### 9) Troubleshooting
- "Invalid number of channels": choose a different `--output-device` index or try `--sample-rate 48000`. The CLI auto-switches mono/stereo, but some devices enforce specific modes.
- "Failed to obtain connection URL": check that `uvicorn` is running and `XI_API_KEY`/`AGENT_ID` are set correctly; verify `BACKEND_URL`.
- No audio playback: confirm your `--output-device` index and system volume; try different devices.
- Microphone not capturing: set `--input-device` correctly and allow mic permissions in your OS.
- Network issues: corporate proxies/firewalls may block WSS; try a different network.

### 10) References
- Quickstart: [ElevenLabs Agents Quickstart](https://elevenlabs.io/docs/agents-platform/quickstart)
- WebSocket API: [Agents Platform WebSocket](https://elevenlabs.io/docs/agents-platform/api-reference/agents-platform/websocket)


