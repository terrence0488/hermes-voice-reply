"""Answer voice notes with a spoken script. See README.md."""
from . import hooks


def register(ctx):
    ctx.register_hook("pre_gateway_dispatch", hooks.on_pre_gateway_dispatch)
    ctx.register_hook("pre_llm_call", hooks.on_pre_llm_call)
    ctx.register_hook("pre_tool_call", hooks.on_pre_tool_call)
    ctx.register_hook("post_tool_call", hooks.on_post_tool_call)
    ctx.register_hook("transform_llm_output", hooks.on_transform_llm_output)
