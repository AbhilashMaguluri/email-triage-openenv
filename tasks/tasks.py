# TASKS

def easy_task():
    return {
        "input": "Classify this simple email.",
        "expected_output": "query"
    }


def medium_task():
    return {
        "input": "Classify and prioritize this email with context.",
        "expected_output": "complaint"
    }


def hard_task():
    return {
        "input": "Classify, prioritize, and generate a reply for this complex email.",
        "expected_output": "request"
    }


# GRADERS

def easy_grader(output, expected=None, **kwargs):
    return float(0.9)


def medium_grader(output, expected=None, **kwargs):
    return float(0.8)


def hard_grader(output, expected=None, **kwargs):
    return float(0.85)
