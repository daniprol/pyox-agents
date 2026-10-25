// k6 load test for server.py: VUS virtual users each send POST /chat in a loop for DURATION.
// Each request reads the whole streamed answer and checks it. Run by load_test.py:
//
//     k6 run -e URL=http://127.0.0.1:8000/chat -e PROMPT="..." -e EXPECTED="..." -e VUS=10 load.js
//
// Two timings per request (k6 built-in metrics):
//   http_req_waiting   time to first byte: until the first piece of the answer arrives
//   http_req_duration  until the whole answer has arrived

import http from "k6/http";
import { check } from "k6";

export const options = {
  vus: Number(__ENV.VUS),
  duration: __ENV.DURATION || "20s",
  summaryTrendStats: ["p(50)", "p(90)", "p(99)", "max"],
};

const body = JSON.stringify({ prompt: __ENV.PROMPT });
const params = { headers: { "Content-Type": "application/json" }, timeout: "60s" };

export default function () {
  const response = http.post(__ENV.URL, body, params);
  check(response, {
    "full answer": (r) => r.status === 200 && r.body.startsWith(__ENV.EXPECTED),
  });
}

// Write the summary as JSON for load_test.py (and skip k6's long text summary).
export function handleSummary(data) {
  return { [__ENV.SUMMARY_FILE]: JSON.stringify(data) };
}
