import subprocess
import sys

def run_fastapi_server():
    # Command to run FastAPI app using Uvicorn
    command = [sys.executable, "-m", "uvicorn", "app.main:app", "--reload", "--port", "9000"]

    # Run the command
    subprocess.run(command)

if __name__ == "__main__":
    run_fastapi_server()
