from pathlib import Path
from streamlit.web import cli

if __name__ == "__main__":
    app_path = str(Path(__file__).parent.resolve() / "app.py")

    cli.main_run([
        app_path,
        "--server.address=localhost",
        #"--server.address=172.20.0.170",
        #"--server.port=8501",
        "--server.port=8502",
        #"--browser.serverAddress=www.clarin.si",
        #"--browser.serverAddress=beta.clarin.si",
        #"--server.baseUrlPath=classla-llm-dashboard",
        #"--server.headless=true",
        #"--browser.gatherUsageStats=false",
        #"--server.showEmailPrompt=false",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false",
        "--server.enableWebsocketCompression=false"
    ])