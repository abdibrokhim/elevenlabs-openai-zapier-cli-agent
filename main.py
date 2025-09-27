import os
import sys
import json
import base64
import signal
import asyncio
import argparse
import threading
import queue
from dataclasses import dataclass, field

import numpy as np
import httpx
import sounddevice as sd
import websockets
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from typing import Any


DEFAULT_SAMPLE_RATE = 16000


@dataclass
class UIState:
    connected: bool = False
    speaking: bool = False
    vad_score: float | None = None
    last_user_transcript: str = ""
    last_agent_response: str = ""
    last_tentative_response: str = ""
    events_seen: int = 0
    errors_seen: int = 0
    info_line: str = ""
    mic_streaming: bool = True


class AudioPlayer:
    def __init__(self, sample_rate: int, device: int | None = None):
        self.sample_rate = sample_rate
        self.device = device
        self.queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=64)
        self._stop = threading.Event()
        self.stream: sd.OutputStream | None = None
        self.thread: threading.Thread | None = None
        self.channels: int = 1

    def start(self):
        if self.stream is not None:
            return
        # Determine supported channel count
        try:
            dev_index = self.device if self.device is not None else sd.default.device[1]
            dev_info = sd.query_devices(dev_index)
            max_ch = int(dev_info.get("max_output_channels", 2) or 2)
        except Exception:
            max_ch = 2
        self.channels = 2 if max_ch >= 2 else 1
        try:
            sd.check_output_settings(device=self.device, samplerate=self.sample_rate, channels=self.channels, dtype="int16")
        except Exception:
            # Flip mono/stereo if initial choice not supported
            self.channels = 1 if self.channels == 2 else 2
            try:
                sd.check_output_settings(device=self.device, samplerate=self.sample_rate, channels=self.channels, dtype="int16")
            except Exception:
                # Keep going; stream.start() may still inform better error
                pass
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            blocksize=0,
            device=self.device,
        )
        self.stream.start()
        self.thread = threading.Thread(target=self._run, name="AudioPlayer", daemon=True)
        self.thread.start()

    def _run(self):
        assert self.stream is not None
        while not self._stop.is_set():
            try:
                data = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                # data is int16 mono; duplicate to match output channels if needed
                if self.channels == 1:
                    frames = data.reshape((-1, 1))
                else:
                    mono = data.reshape((-1, 1))
                    frames = np.repeat(mono, self.channels, axis=1)
                self.stream.write(frames)
            except Exception:
                # swallow audio glitches
                pass

    def play_int16(self, pcm_int16: np.ndarray):
        if self.stream is None:
            self.start()
        try:
            self.queue.put_nowait(pcm_int16)
        except queue.Full:
            # drop if overwhelmed
            self.queue.get_nowait()
            self.queue.put_nowait(pcm_int16)

    def stop(self):
        self._stop.set()
        if self.thread:
            self.thread.join(timeout=0.5)
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
        self.stream = None


class MicRecorder:
    def __init__(self, sample_rate: int, device: int | None = None, enabled: bool = True):
        self.sample_rate = sample_rate
        self.device = device
        self.enabled = enabled
        self.queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=64)
        self.stream: sd.InputStream | None = None
        self.channels: int = 1

    def _callback(self, indata, frames, time_info, status):
        if status:
            # Capture glitches are not fatal. Ignore.
            pass
        if not self.enabled:
            return
        try:
            # indata dtype is float32 by default; convert to int16 PCM range
            floats = np.copy(indata[:, 0])  # mono
            clipped = np.clip(floats, -1.0, 1.0)
            pcm_int16 = (clipped * 32767.0).astype(np.int16)
            self.queue.put_nowait(pcm_int16)
        except queue.Full:
            try:
                _ = self.queue.get_nowait()
            except queue.Empty:
                pass

    def start(self):
        if not self.enabled or self.stream is not None:
            return
        # Choose input channels with a simple fallback
        ch = 1
        try:
            sd.check_input_settings(device=self.device, samplerate=self.sample_rate, channels=ch, dtype="float32")
        except Exception:
            ch = 2
        self.channels = ch
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=1600,
            device=self.device,
            callback=self._callback,
        )
        self.stream.start()

    def stop(self):
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
        self.stream = None


def render_ui(state: UIState) -> Panel:
    status = Table.grid(expand=True)
    status.add_column(justify="left")
    status.add_column(justify="right")
    conn_text = "Connected" if state.connected else "Disconnected"
    mode_text = "Speaking" if state.speaking else "Listening"
    vad = f"{state.vad_score:.2f}" if state.vad_score is not None else "-"
    status.add_row(f"Status: [bold]{conn_text}[/]  |  Mode: [bold]{mode_text}[/]  |  VAD: {vad}", f"events: {state.events_seen}  errors: {state.errors_seen}")

    convo = Table.grid(padding=(0, 1))
    convo.add_column(width=12, style="cyan", no_wrap=True)
    convo.add_column()
    if state.last_user_transcript:
        convo.add_row("You", Text(state.last_user_transcript))
    if state.last_tentative_response:
        convo.add_row("Agent (draft)", Text(state.last_tentative_response, style="yellow"))
    if state.last_agent_response:
        convo.add_row("Agent", Text(state.last_agent_response, style="green"))

    footer = Text()
    footer.append("Enter to type a message. ")
    footer.append("Ctrl+C to quit. ")
    footer.append("Mic: ")
    footer.append("ON" if state.mic_streaming else "OFF", style="bold")
    if state.info_line:
        footer.append(f"  |  {state.info_line}")

    outer = Table.grid(expand=True)
    outer.add_row(status)
    outer.add_row(convo)
    outer.add_row(footer)
    return Panel(outer, title="CLI Conversational Agent", border_style="magenta")


async def fetch_signed_url(backend_url: str, console: Console) -> str:
    url = backend_url.rstrip("/") + "/api/signed-url"
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(url)
        r.raise_for_status()
        data = r.json()
        if "signedUrl" not in data:
            raise RuntimeError("signedUrl missing in response")
        console.log("Obtained signed URL from backend")
        return data["signedUrl"]


async def send_conversation_initiation(ws: Any):
    payload = {
        "type": "conversation_initiation_client_data",
        "conversation_config_override": {"agent": {}},
        "custom_llm_extra_body": {},
        "dynamic_variables": {},
    }
    await ws.send(json.dumps(payload))


async def mic_sender_task(ws: Any, mic: MicRecorder, state: UIState, stop_event: asyncio.Event):
    loop = asyncio.get_running_loop()
    while not stop_event.is_set():
        try:
            pcm_chunk: np.ndarray = await asyncio.to_thread(mic.queue.get, True, 0.1)
        except queue.Empty:
            await asyncio.sleep(0.01)
            continue
        if not state.mic_streaming:
            continue
        # pcm_chunk is int16 mono
        try:
            b = pcm_chunk.tobytes()
            b64 = base64.b64encode(b).decode("ascii")
            await ws.send(json.dumps({"user_audio_chunk": b64}))
        except Exception:
            # Non-fatal send error; likely connection closing
            await asyncio.sleep(0.01)


async def keyboard_task(ws: Any, state: UIState, stop_event: asyncio.Event, console: Console):
    while not stop_event.is_set():
        try:
            line = await asyncio.to_thread(sys.stdin.readline)
        except Exception:
            break
        if not line:
            await asyncio.sleep(0.1)
            continue
        text = line.strip()
        if not text:
            continue
        try:
            await ws.send(json.dumps({"type": "user_message", "text": text}))
            state.info_line = "Sent text message"
        except Exception:
            state.errors_seen += 1
            state.info_line = "Failed to send text message"


async def receiver_task(ws: Any, player: AudioPlayer, state: UIState, stop_event: asyncio.Event):
    while not stop_event.is_set():
        try:
            raw = await ws.recv()
        except websockets.ConnectionClosed:
            break
        except Exception:
            state.errors_seen += 1
            continue
        try:
            msg = json.loads(raw)
        except Exception:
            state.errors_seen += 1
            continue
        state.events_seen += 1
        msg_type = msg.get("type")
        if msg_type == "conversation_initiation_metadata":
            pass
        elif msg_type == "vad_score":
            ev = msg.get("vad_score_event", {})
            state.vad_score = ev.get("vad_score")
        elif msg_type == "user_transcript":
            ev = msg.get("user_transcription_event", {})
            state.last_user_transcript = ev.get("user_transcript", "")
        elif msg_type == "internal_tentative_agent_response":
            ev = msg.get("tentative_agent_response_internal_event", {})
            state.last_tentative_response = ev.get("tentative_agent_response", "")
        elif msg_type == "agent_response":
            ev = msg.get("agent_response_event", {})
            state.last_agent_response = ev.get("agent_response", "")
            state.last_tentative_response = ""
        elif msg_type == "audio":
            ev = msg.get("audio_event", {})
            b64 = ev.get("audio_base_64") or ev.get("audio_base64")
            if b64:
                try:
                    b = base64.b64decode(b64)
                    pcm = np.frombuffer(b, dtype=np.int16)
                    player.play_int16(pcm)
                    state.speaking = True
                except Exception:
                    state.errors_seen += 1
            else:
                state.speaking = False
        elif msg_type == "ping":
            ev = msg.get("ping_event", {})
            event_id = ev.get("event_id")
            try:
                await ws.send(json.dumps({"type": "pong", "event_id": event_id}))
            except Exception:
                pass
        elif msg_type == "client_tool_call":
            ev = msg.get("client_tool_call", {})
            tool_call_id = ev.get("tool_call_id")
            try:
                await ws.send(json.dumps({
                    "type": "client_tool_result",
                    "tool_call_id": tool_call_id,
                    "result": "No client tools available in CLI",
                    "is_error": True,
                }))
            except Exception:
                pass
        else:
            # Unknown types are ignored
            pass


async def ui_task(state: UIState, stop_event: asyncio.Event, console: Console):
    with Live(render_ui(state), console=console, refresh_per_second=20, screen=False) as live:
        while not stop_event.is_set():
            await asyncio.sleep(0.05)
            # reset speaking flag if no audio recently; heuristic
            state.speaking = state.speaking and (state.vad_score is not None and state.vad_score < 0.99)
            live.update(render_ui(state))


async def run_client(args: argparse.Namespace):
    console = Console()
    state = UIState(mic_streaming=not args.no_mic)
    player = AudioPlayer(sample_rate=args.sample_rate, device=args.output_device)
    mic = MicRecorder(sample_rate=args.sample_rate, device=args.input_device, enabled=not args.no_mic)

    # Graceful shutdown handling
    stop_event = asyncio.Event()

    def _signal_handler(*_):
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            signal.signal(sig, lambda *_: stop_event.set())

    backend_url = args.backend_url or os.getenv("BACKEND_URL", "http://localhost:8000")
    signed_url: str
    try:
        if args.agent_id:
            signed_url = f"wss://api.elevenlabs.io/v1/convai/conversation?agent_id={args.agent_id}"
        else:
            signed_url = await fetch_signed_url(backend_url, console)
    except Exception as e:
        console.print(f"[red]Failed to obtain connection URL:[/] {e}")
        return

    try:
        async with websockets.connect(signed_url, ping_interval=None, max_size=10_000_000) as ws:
            state.connected = True
            # Start audio IO
            player.start()
            mic.start()

            await send_conversation_initiation(ws)

            recv = asyncio.create_task(receiver_task(ws, player, state, stop_event))
            tasks = [recv]
            if not args.no_mic:
                tasks.append(asyncio.create_task(mic_sender_task(ws, mic, state, stop_event)))
            tasks.append(asyncio.create_task(keyboard_task(ws, state, stop_event, console)))
            tasks.append(asyncio.create_task(ui_task(state, stop_event, console)))

            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            stop_event.set()
            for t in tasks:
                t.cancel()
            try:
                await ws.close()
            except Exception:
                pass
    except Exception as e:
        state.errors_seen += 1
        Console().print(f"[red]Connection error:[/] {e}")
    finally:
        state.connected = False
        mic.stop()
        player.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLI AI Conversational Agent (ElevenLabs)")
    parser.add_argument("--backend-url", dest="backend_url", default=os.getenv("BACKEND_URL"), help="Backend base URL providing /api/signed-url (default: http://localhost:8000)")
    parser.add_argument("--agent-id", dest="agent_id", default=os.getenv("AGENT_ID"), help="Public Agent ID (skips signed URL)")
    parser.add_argument("--sample-rate", dest="sample_rate", type=int, default=DEFAULT_SAMPLE_RATE, help="Audio sample rate (Hz)")
    parser.add_argument("--input-device", dest="input_device", type=int, default=None, help="Sounddevice input device index")
    parser.add_argument("--output-device", dest="output_device", type=int, default=None, help="Sounddevice output device index")
    parser.add_argument("--no-mic", action="store_true", help="Disable microphone streaming; type messages only")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        args = parse_args()
        asyncio.run(run_client(args))
    except KeyboardInterrupt:
        pass

