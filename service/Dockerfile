FROM adefe/strawberry_env:v3

WORKDIR /home

COPY . /home

CMD ["uvicorn", "--app-dir", "src", "--host", "0.0.0.0", "--port", "20002", "server:app"]
