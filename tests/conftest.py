import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--milvus-uri",
        default="http://localhost:19530",
        help="Milvus connection URI",
    )


@pytest.fixture(scope="session")
def milvus_uri(request):
    return request.config.getoption("--milvus-uri")
