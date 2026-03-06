from vlmeval.api import OpenAIWrapper

if __name__ == "__main__":
    model = OpenAIWrapper("gpt-4o-mini", verbose=True)
    msgs = [dict(type="text", value="Hello!")]
    code, answer, resp = model.generate_inner(msgs)
    print(code, answer, resp)
