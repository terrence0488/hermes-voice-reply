# hermes-voice-reply

Makes a [Hermes Agent](https://github.com/NousResearch/hermes-agent) answer a voice note with a spoken script, not a written
answer read aloud. Tables, code and other detail stay in the text. Typed messages stay text.

Needs the Fish Audio plugin (`hermes plugins install fish-audio`), set up as the speech provider. Leave `/voice` off: it reads the final text aloud.

## How it works

Hermes hands the model a voice note as a plain quoted transcript, so the model can't reliably tell it was spoken.

| Hook | Job |
|---|---|
| `pre_gateway_dispatch` | Sees the real message type, and notes a voice note from that person, in a direct message |
| `pre_llm_call` | On that turn only, adds the instruction next to the message |
| `pre_tool_call` | On that turn, refuses a script with code, tables, JSON, HTML, URLs, file paths, markdown or over 110 words, so the model rewrites it |
| `pre_tool_call`, always | Replaces any `model` or `voice` the model chose in `fish_speak` with your `tts.fish-audio` settings, unless your message named that value. Models copy these from memory, which can bill a paid model or use an old voice |
| `pre_tool_call`, typed turns | Refuses speech unless the message asks for it (`voice`, `say`, `speak`, `aloud`...), because models copy voice replies from earlier in the chat |
| `post_tool_call` | Records the audio file the tool made |
| `transform_llm_output` | If no real audio exists, removes any made-up `MEDIA:` line and makes the audio itself |

The fallback speaks a short plain reply as written. Otherwise it says the details are in the text. If that fails too, the text gets a short note.

## Install

```bash
hermes plugins install "https://github.com/terrence0488/hermes-voice-reply" --ref <commit>
```

```bash
hermes plugins enable voice_reply
```

Restart the gateway.

## Limits

- **Direct messages only.** Voice notes in groups are ignored.
- **The fallback sounds plainer** than a script the model wrote, because no one wrote it for speech.
- **Asking for voice is a word match.** A typed message with `say` in it, such as "what did he say", is allowed to use voice.
- **The guard catches shapes, not meaning.** A long spoken list in plain sentences under 110 words passes.
- **The flag matches by person, not message.** If someone sends a voice note and a typed message close together, the first turn to start takes it.
- **It calls `fish_speak` by name,** so it depends on the Fish Audio plugin's tool.
- **Everything spoken and transcribed goes to Fish Audio.**

## Tests

```bash
python3 -m unittest discover -s tests
```

These use stand-ins for Hermes. Check real behaviour with live voice notes and typed messages.

## Licence

MIT
