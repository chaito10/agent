from agent import Agent
import os

# Try to connect to llama.cpp server
agent = Agent(name="test", root=".", model="model.llamacpp")
# Default is http://127.0.0.1:8080
try:
    result = agent.call("model.llamacpp", {
        "messages": [{"role": "user", "content": "Hello, respond with 'hi'"}],
        "goal": "test",
        "step": 0,
        "temperature": 0.1,
        "n_predict": 50
    })
    print(result.value)
except Exception as e:
    print(f"ERR: {e}")
    import traceback
    traceback.print_exc()
