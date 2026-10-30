// Reads a stream made by pyox-agents and checks it with the official client library.
//
//   node check.mjs ai-sdk|ai-sdk-6|ag-ui < input.json
//
// input.json: {"sse": "<the stream text>", "message": <AI SDK: the UIMessage to continue, or null>}
// Prints JSON: for the AI SDK, `{message, errors}`: the UIMessage the client built
// from the stream and the errors it reported; for AG-UI, the events that passed
// `verifyEvents` (which throws on a wrong order).
import { readFileSync } from "node:fs";
import { verifyEvents } from "@ag-ui/client";
import { from, lastValueFrom, toArray } from "rxjs";
import * as ai7 from "ai";
import * as ai6 from "ai-6";

const protocol = process.argv[2];
const input = JSON.parse(readFileSync(0, "utf8"));

// "data: {...}" lines to objects (both protocols use server-sent events).
const chunks = input.sse
  .split("\n\n")
  .filter((event) => event.startsWith("data: ") && event !== "data: [DONE]")
  .map((event) => JSON.parse(event.slice("data: ".length)));

if (protocol === "ag-ui") {
  const events = await lastValueFrom(from(chunks).pipe(verifyEvents(false), toArray()));
  console.log(JSON.stringify(events));
} else {
  const ai = protocol === "ai-sdk-6" ? ai6 : ai7;
  const schema = ai.uiMessageChunkSchema();
  for (const chunk of chunks) {
    const result = await schema.validate(chunk);
    if (!result.success) throw new Error(`Invalid chunk ${JSON.stringify(chunk)}: ${result.error}`);
  }
  const stream = new ReadableStream({
    start(controller) {
      chunks.forEach((chunk) => controller.enqueue(chunk));
      controller.close();
    },
  });
  const errors = [];
  const onError = (error) => errors.push(error.message);
  let message = null;
  for await (message of ai.readUIMessageStream({ stream, message: input.message ?? undefined, onError })) {
  }
  console.log(JSON.stringify({ message, errors }));
}
