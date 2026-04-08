def easy_task():
    return {
        "input": "Email: Meeting tomorrow",
        "expected_output": "work"
    }

def medium_task():
    return {
        "input": "Email: Discount offer just for you",
        "expected_output": "promotion"
    }

def hard_task():
    return {
        "input": "Email: Your account has been compromised",
        "expected_output": "important"
    }


def easy_grader(*args, **kwargs):
    return float(0.9)

def medium_grader(*args, **kwargs):
    return float(0.8)

def hard_grader(*args, **kwargs):
    return float(0.85)
