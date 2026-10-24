import os
from dotenv import load_dotenv
from openai import OpenAI
from agent import ResearchAgent


def main():
    # 加载.env
    load_dotenv()
    # 获取 api_key、model、base_url
    api_key = os.getenv('OPENAI_API_KEY')
    model = os.getenv('MODEL')
    base_url = os.getenv('OPENAI_BASE_URL')

    if not api_key or not model:
        raise SystemExit('Set OPENAI_API_KEY and MODEL_NAME in .env')
    
    # 实例化OpenAI客户端    
    client = OpenAI(api_key=api_key, base_url=base_url or None, timeout=30.0, max_retries=1)
    # 实例化一个Agent
    agent = ResearchAgent(client, model=model)
    print('Research Agent L0 | enter exit to quit')
    while True:
        try:
            task = input('\nYou> ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\nBye')
            break
        if task.lower() in {'exit', 'quit'}:
            break
        if not task:
            continue
        try:
            print('\nAgent>', agent.run(task))
        except Exception as exc:
            print(f'API request or runtime error: {exc}')


if __name__ == '__main__':
    main()
