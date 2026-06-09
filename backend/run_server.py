#!/usr/bin/env python3
"""
GPT-Researcher Backend Server Startup Script

Run this to start the research API server.
"""

import uvicorn
import os
import sys

# Add the backend directory to Python path (so `server.app` is importable)
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

# The `gpt_researcher` package lives at the project root, one level above
# backend/. Put the project root on PYTHONPATH so it is importable both here
# and in the subprocess uvicorn spawns when reload=True (the reloaded worker
# inherits env vars, not sys.path).
project_root = os.path.dirname(backend_dir)
sys.path.insert(0, project_root)
existing_pythonpath = os.environ.get("PYTHONPATH", "")
if project_root not in existing_pythonpath.split(os.pathsep):
    os.environ["PYTHONPATH"] = (
        project_root + os.pathsep + existing_pythonpath if existing_pythonpath else project_root
    )

if __name__ == "__main__":
    # Change to backend directory
    os.chdir(backend_dir)

    # Start the server
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )



