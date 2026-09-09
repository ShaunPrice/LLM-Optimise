"""Read-only MCP smoke client; requires pip install '.[mcp]'."""

import argparse
import asyncio
import json
import os

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


async def run(url):
    token = os.environ.get("LLM_OPTIMISE_MCP_TOKEN")
    if not token:
        raise SystemExit("Set LLM_OPTIMISE_MCP_TOKEN in this client's environment")
    async with (
        httpx.AsyncClient(
            headers={"Authorization": "Bearer " + token}, trust_env=False
        ) as http_client,
        streamable_http_client(url, http_client=http_client) as (read, write, _),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        print(
            json.dumps(
                {"tools": [tool.name for tool in (await session.list_tools()).tools]}, indent=2
            )
        )
        for name, arguments in (("discover", {"action_id": "route"}), ("hardware", {})):
            result = await session.call_tool(name, arguments)
            print(result.model_dump_json(indent=2))
            if result.isError:
                raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8766/mcp")
    asyncio.run(run(parser.parse_args().url))
