import json
import sys
from collections import Counter
from pathlib import Path


def main() -> None:
    path = Path(sys.argv[1])
    data = json.loads(path.read_text(encoding="utf-8"))

    events = data.get("events", [])
    types = Counter(e["type"] for e in events)

    print(f"Transcript: {path.name}")
    print(f"Total events: {len(events)}")
    print()
    print("Event type counts:")
    for t, n in types.most_common():
        print(f"  {t:30s} {n}")
    print()

    # Token usage totals
    in_tok = sum(e.get("input_tokens", 0) for e in events if e["type"] == "token_usage")
    out_tok = sum(e.get("output_tokens", 0) for e in events if e["type"] == "token_usage")
    cache_read = sum(e.get("cache_read_tokens") or 0 for e in events if e["type"] == "token_usage")
    cache_write = sum(e.get("cache_write_tokens") or 0 for e in events if e["type"] == "token_usage")
    print(f"Total tokens: in={in_tok}  out={out_tok}  cache_read={cache_read}  cache_write={cache_write}")
    print()

    # Tool call analysis
    tool_calls = [e for e in events if e["type"] == "tool_call_started"]
    print(f"Tool calls: {len(tool_calls)}")
    if tool_calls:
        # Categorize bash calls
        empty_calls = 0
        valid_calls = 0
        restart_calls = 0
        for e in tool_calls:
            args_str = e.get("tool_call", {}).get("function", {}).get("arguments", "")
            if args_str == "{}" or args_str == "":
                empty_calls += 1
            else:
                try:
                    args = json.loads(args_str)
                    if args.get("restart"):
                        restart_calls += 1
                    elif args.get("command"):
                        valid_calls += 1
                    else:
                        empty_calls += 1
                except json.JSONDecodeError:
                    empty_calls += 1
        print(f"  empty/no-command: {empty_calls}")
        print(f"  valid command:    {valid_calls}")
        print(f"  restart:          {restart_calls}")
    print()

    # Print first N valid bash commands the agent tried
    print("First 10 valid bash commands the agent tried:")
    n_shown = 0
    for e in tool_calls:
        args_str = e.get("tool_call", {}).get("function", {}).get("arguments", "")
        try:
            args = json.loads(args_str)
            cmd = args.get("command")
            if cmd:
                print(f"  [{n_shown+1}] {cmd[:200]}{'...' if len(cmd) > 200 else ''}")
                n_shown += 1
                if n_shown >= 10:
                    break
        except json.JSONDecodeError:
            pass
    print()

    # First few assistant messages (model's actual reasoning text)
    print("First 5 assistant message contents (model thinking text):")
    n_shown = 0
    for e in events:
        if e["type"] == "message_added":
            msg = e["message"]
            if msg.get("role") == "assistant":
                content = msg.get("content")
                if content:
                    if isinstance(content, list):
                        for chunk in content:
                            if isinstance(chunk, dict) and chunk.get("type") == "text":
                                text = chunk.get("text", "")
                                if text.strip():
                                    print(f"  [{n_shown+1}] {text[:300]}{'...' if len(text) > 300 else ''}")
                                    n_shown += 1
                                    break
                    elif isinstance(content, str) and content.strip():
                        print(f"  [{n_shown+1}] {content[:300]}{'...' if len(content) > 300 else ''}")
                        n_shown += 1
                if n_shown >= 5:
                    break
    print()

    # Final status
    completed = [e for e in events if e["type"] == "task_completed"]
    if completed:
        print(f"Task status: {completed[-1].get('status')}")
    errors = [e for e in events if e["type"] == "error"]
    if errors:
        print(f"Errors: {len(errors)}")
        for e in errors[:3]:
            print(f"  {e.get('exception_type')}: {e.get('message')[:200]}")

    # Look at what's actually stored in failed tool calls
    print()
    print("Sample of empty tool_call records (raw arguments string):")
    n_shown = 0
    for e in tool_calls:
        args_str = e.get("tool_call", {}).get("function", {}).get("arguments", "")
        if args_str in ("{}", "", None):
            print(f"  arguments=={args_str!r}")
            n_shown += 1
            if n_shown >= 5:
                break

    # And: what does the assistant message LOOK like for an empty call?
    # Find the tool_call_started, then the message_added preceding it
    print()
    print("Assistant message right before first empty tool call:")
    for i, e in enumerate(events):
        if e["type"] == "tool_call_started":
            args_str = e.get("tool_call", {}).get("function", {}).get("arguments", "")
            if args_str == "{}":
                # Find the most recent message_added with role=assistant before this
                for j in range(i - 1, -1, -1):
                    if events[j]["type"] == "message_added":
                        msg = events[j]["message"]
                        if msg.get("role") == "assistant":
                            print(json.dumps(msg, indent=2)[:2000])
                            break
                break


if __name__ == "__main__":
    main()
