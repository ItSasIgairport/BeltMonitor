FROM belt-monitor-base:latest

WORKDIR /app
COPY . /app

# Install remaining requirements (if any new ones are added)
# We try to install from local 'wheels' folder if present, otherwise skip or try online
RUN if [ -f requirements.txt ]; then \
    pip install --no-index --find-links=/home/havaciliksistemleri/BeltMonitor/wheels -r requirements.txt || echo "Requirements install failed or skipped"; \
    fi

# Startup Command
CMD ["sh", "-c", "service ssh start && python3 main.py"]
