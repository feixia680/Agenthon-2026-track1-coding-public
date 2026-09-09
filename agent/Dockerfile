FROM finance-bench-sandbox:latest

RUN pip install --no-cache-dir openai==1.109.1

COPY agent.py /agent/agent.py

LABEL qfbench2.interface_version="2.0"

ENTRYPOINT ["python", "/agent/agent.py"]
