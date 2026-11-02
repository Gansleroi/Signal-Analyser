# conftest.py
def pytest_runtest_logreport(report):
    if report.when == "call":
        status = "PASSED" if report.passed else "FAILED"

        with open("test_results.log", "a", encoding="utf-8") as f:
            f.write(f"{status}: {report.nodeid} (Длительность: {report.duration:.3f}с)\n")