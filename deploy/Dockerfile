FROM belt-monitor-base:latest

WORKDIR /app
COPY . /app

# Install remaining requirements (if any new ones are added)
# We try to install from local 'wheels' folder if present, otherwise skip or try online
RUN pip install --no-index --find-links=wheels -r requirements.txt

# Startup Command
CMD ["sh", "-c", "service ssh start && python3 main.py"]