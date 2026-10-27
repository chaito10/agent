from agent import Agent

agent = Agent(name="test", root=".", allow_network=True)
result = agent.call("model.llamacpp", {
    "messages": [{"role": "system", "content": "You are helpful"}, {"role": "user", "content": "Say hi"}],
    "goal": "test",
    "step": 0,
    "base_url": "http://127.0.0.1:8080",
    "temperature": 0.1,
    "n_predict": 30
})
print(type(result))
print(result.ok, result.blocked)
print(result.error)
print(result.value)
