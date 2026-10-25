"""Ask an agent about an image and a PDF.

OPENAI_API_KEY=... uv run examples/images_and_files.py path/to/some.pdf
"""

import asyncio
import sys

from pyox_agents import Agent, file, image

CAT_PHOTO = "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg"

agent = Agent(model="gpt-5-mini", prompt="Be brief.")


async def main(pdf_path: str) -> None:
    # A web address is sent as is: the API downloads the image.
    result = await agent.run(["What animal is this?", image(CAT_PHOTO)])
    print(result.text)

    # A local file is read and sent inline. `image("photo.png")` works the same way.
    result = await agent.run(["Summarize this document in one sentence.", file(pdf_path)])
    print(result.text)


asyncio.run(main(sys.argv[1]))
