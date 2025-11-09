def get_llm_response(query: str, tool_name=None) -> str:
    try:
        agent = get_agent()
        result = agent.invoke({"messages": [{"role": "user", "content": query}]})

        # ✅ Case 1: LangChain-like dict response
        if isinstance(result, dict) and "messages" in result:
            messages = result["messages"]
            if isinstance(messages, list) and len(messages) > 0:
                last_msg = messages[-1]
                if hasattr(last_msg, "content"):
                    return str(last_msg.content)
                elif isinstance(last_msg, dict) and last_msg.get("content"):
                    return last_msg["content"]

        # ✅ Case 2: result is list (rare in Groq agents)
        if isinstance(result, list):
            # Flatten to string
            combined = " ".join(
                [
                    msg.content if hasattr(msg, "content") else str(msg)
                    for msg in result
                ]
            )
            return combined

        # ✅ Case 3: it's already a clean string
        if isinstance(result, str):
            return result

        # ✅ Catch-all for unexpected result types
        return str(result)

    except Exception as e:
        return f"🤖 क्षमा करें! तकनीकी समस्या के कारण जवाब तैयार नहीं हो सका। ({str(e)})"
