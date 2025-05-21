from llm_api_wrapper import basic_prompt


basic_prompt("What is the capital of Germany?", "Make up a funny answer that is incorrect.", debug=True,
             model="gemini-2.0-flash")