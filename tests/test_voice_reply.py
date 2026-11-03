"""Plugin tests with stand-ins for Hermes. Real behaviour is checked live, see README.md."""
import importlib.util
import json
import os
import pathlib
import sys
import tempfile
import types
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("voice_reply", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
plugin = importlib.util.module_from_spec(spec)
sys.modules["voice_reply"] = plugin
spec.loader.exec_module(plugin)
hooks = sys.modules["voice_reply.hooks"]
check = sys.modules["voice_reply.script_check"]


def event(kind="VOICE", user="42", chat_type="dm", internal=False):
    source = types.SimpleNamespace(platform=types.SimpleNamespace(value="telegram"), user_id=user, chat_type=chat_type)
    return types.SimpleNamespace(message_type=types.SimpleNamespace(name=kind), source=source, internal=internal)


class Base(unittest.TestCase):
    def setUp(self):
        hooks._pending.clear()
        hooks._turns.clear()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.audio = os.path.join(self.tmp.name, "real.ogg")
        open(self.audio, "w").close()

    def start(self, **kw):
        hooks.on_pre_gateway_dispatch(event=event(**kw))
        return hooks.on_pre_llm_call(session_id="s", task_id="t", platform="telegram", sender_id="42")


class Detection(Base):
    def test_voice_note_gets_the_instruction(self):
        self.assertIn("fish_speak", self.start()["context"])

    def test_typed_message_gets_nothing(self):
        self.assertIsNone(self.start(kind="TEXT"))

    def test_quoted_typed_message_gets_nothing(self):
        self.assertIsNone(hooks.on_pre_llm_call(session_id="s", platform="telegram", sender_id="42",
                                                user_message='"typed in quotes"'))

    def test_group_voice_is_ignored(self):
        self.assertIsNone(self.start(chat_type="group"))

    def test_internal_event_is_ignored(self):
        self.assertIsNone(self.start(internal=True))

    def test_flag_is_used_once(self):
        self.assertTrue(self.start())
        self.assertIsNone(hooks.on_pre_llm_call(session_id="s2", platform="telegram", sender_id="42"))

    def test_another_user_is_not_affected(self):
        hooks.on_pre_gateway_dispatch(event=event(user="42"))
        self.assertIsNone(hooks.on_pre_llm_call(session_id="s", platform="telegram", sender_id="99"))

    def test_old_flag_expires(self):
        hooks.on_pre_gateway_dispatch(event=event())
        hooks._pending[("telegram", "42")] = [-hooks.PENDING_TTL_SECONDS - 1e9]
        self.assertIsNone(hooks.on_pre_llm_call(session_id="s", platform="telegram", sender_id="42"))

    def test_strangers_cannot_grow_the_table_without_limit(self):
        for number in range(hooks.MAX_TRACKED + 50):
            hooks.on_pre_gateway_dispatch(event=event(user=str(number)))
        self.assertLessEqual(len(hooks._pending), hooks.MAX_TRACKED)


class Guard(Base):
    def blocked(self, text, wrapper=False):
        self.start()
        if wrapper:
            return hooks.on_pre_tool_call("tool_call", {"calls": [{"name": "fish_speak", "arguments": {"text": text}}]}, "t")
        return hooks.on_pre_tool_call("fish_speak", {"text": text}, "t")

    def test_natural_speech_passes(self):
        self.assertIsNone(self.blocked("[calm] Short version: A's cheaper, but B lasts longer. Table's in the text."))

    def test_bad_scripts_are_blocked(self):
        for text in ("```python\nx = 1\n```", "| a | b |\n|---|---|\n| 1 | 2 |", '{"a": 1}', "<div>hi</div>",
                     "See https://example.com now", "Open /home/me/file.txt", "## Title", "- one\n- two",
                     "word " * 200):
            with self.subTest(text[:20]):
                hooks._turns.clear()
                self.assertEqual(self.blocked(text)["action"], "block")

    def test_wrapped_call_is_checked(self):
        self.assertEqual(self.blocked("```x```", wrapper=True)["action"], "block")

    def test_typed_turn_is_not_guarded(self):
        self.assertIsNone(hooks.on_pre_tool_call("fish_speak", {"text": "```x```"}, "t"))

    def test_other_tools_are_ignored(self):
        self.start()
        self.assertIsNone(hooks.on_pre_tool_call("terminal", {"command": "```"}, "t"))


class Finish(Base):
    def setUp(self):
        super().setUp()
        self.spoken = []
        self.make = lambda text: self.spoken.append(text) or f"[[audio_as_voice]]\nMEDIA:{self.audio}"
        original = hooks._synthesize
        hooks._synthesize = lambda text: self.make(text)
        self.addCleanup(setattr, hooks, "_synthesize", original)

    def finish(self, text):
        return hooks.on_transform_llm_output(response_text=text, session_id="s", task_id="t")

    def test_typed_turn_is_untouched(self):
        self.assertIsNone(self.finish("hello"))

    def test_real_audio_is_untouched(self):
        self.start()
        hooks.on_post_tool_call("fish_speak", {"text": "hi"}, json.dumps({"media_tag": f"MEDIA:{self.audio}"}), "t")
        self.assertIsNone(self.finish("Caption"))
        self.assertEqual(self.spoken, [])

    def test_wrapped_tool_result_counts(self):
        self.start()
        hooks.on_post_tool_call("tool_call", {"calls": [{"name": "fish_speak", "arguments": {"text": "hi"}}]},
                                json.dumps({"results": [{"media_tag": f"MEDIA:{self.audio}"}]}), "t")
        self.assertIsNone(self.finish("Caption"))

    def test_missed_voice_speaks_a_simple_reply(self):
        self.start()
        out = self.finish("Doing fine, thanks.")
        self.assertEqual(self.spoken, ["Doing fine, thanks."])
        self.assertIn(self.audio, out)
        self.assertIn("Doing fine", out)

    def test_missed_voice_with_code_speaks_a_pointer(self):
        self.start()
        out = self.finish("Here:\n```python\nx = 1\n```")
        self.assertEqual(self.spoken, [hooks.FALLBACK_DETAIL_LINE])
        self.assertIn("```python", out)

    def test_faked_media_is_removed_and_replaced(self):
        self.start()
        out = self.finish("[calm] Doing good.\n\nMEDIA:/nope/fake.mp3")
        self.assertNotIn("/nope/fake.mp3", out)
        self.assertIn(self.audio, out)

    def test_model_written_real_media_is_kept(self):
        self.start()
        self.assertIsNone(self.finish(f"Caption\nMEDIA:{self.audio}"))
        self.assertEqual(self.spoken, [])

    def test_failed_fallback_says_so_and_keeps_the_text(self):
        def boom(text):
            raise RuntimeError("no")
        hooks._synthesize = boom
        self.start()
        out = self.finish("Doing fine.")
        self.assertIn("Doing fine.", out)
        self.assertIn(hooks.FAILED_NOTE, out)

    def test_turn_is_closed_after_finishing(self):
        self.start()
        self.finish("Doing fine.")
        self.assertEqual(hooks._turns, {})
        self.assertIsNone(hooks.on_pre_tool_call("fish_speak", {"text": "```x```"}, "t"))


class Failure(Base):
    def test_hooks_never_raise_on_junk(self):
        self.assertIsNone(hooks.on_pre_gateway_dispatch(event=object()))
        self.assertIsNone(hooks.on_pre_llm_call())
        self.assertIsNone(hooks.on_pre_tool_call(None, None))
        self.assertIsNone(hooks.on_post_tool_call(None, None, object()))
        self.assertIsNone(hooks.on_transform_llm_output())


class Registration(unittest.TestCase):
    def test_registers_every_declared_hook(self):
        seen = []
        plugin.register(types.SimpleNamespace(register_hook=lambda name, fn: seen.append(name)))
        manifest = (ROOT / "plugin.yaml").read_text()
        for name in seen:
            self.assertIn(f"- {name}", manifest)
        self.assertEqual(len(seen), 5)


if __name__ == "__main__":
    unittest.main()
