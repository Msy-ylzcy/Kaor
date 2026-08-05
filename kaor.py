from backend.startup_trace import trace_startup


trace_startup("entry point started")

from backend.main import main


trace_startup("backend.main imported")


if __name__ == "__main__":
    trace_startup("calling backend.main.main")
    main()
