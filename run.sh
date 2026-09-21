#!/bin/bash
# Lambda Web Adapter startup script. LWA's wrapper (/opt/bootstrap, from the
# layer) execs this as the Lambda "handler" instead of a Python function --
# it starts a real, long-lived uvicorn process serving the SAME FastAPI
# app object the Mangum-based path already imports (app.main:app). Nothing
# in app/ changes: app.main.handler / _mangum_handler are simply never
# called in this path -- they sit unused, harmlessly (kept so the function
# still works if you ever revert Handler back to app.main.handler).
PATH=$PATH:$LAMBDA_TASK_ROOT/bin \
    PYTHONPATH=$PYTHONPATH:/opt/python:$LAMBDA_RUNTIME_DIR \
    exec python -m uvicorn --host 0.0.0.0 --port "$PORT" app.main:app