import os
import sys

from dotenv import load_dotenv

from uipath_langchain._cli.cli_run import langgraph_run_middleware

load_dotenv()


def test_dummy():
    test_folder_path = os.path.dirname(os.path.abspath(__file__))
    sample_path = os.path.join(test_folder_path, "samples", "1-simple-graph")

    sys.path.append(sample_path)
    os.chdir(sample_path)
    result = langgraph_run_middleware(
        entrypoint=None,
        input='{ "graph_state": "GET Assets API does not enforce proper permissions Assets.View" }',
        resume=False,
    )

    assert result.error_message is None
