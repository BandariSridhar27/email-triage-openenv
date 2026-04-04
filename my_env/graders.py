def grade_easy(action, truth):
    return 1.0 if action.classification == truth["label"] else 0.0


def grade_medium(action, truth):
    score = 0.0

    if action.classification == truth["label"]:
        score += 0.5

    if action.priority == truth["priority"]:
        score += 0.5

    return score


def grade_hard(action, truth):
    score = 0.0

    if action.classification == truth["label"]:
        score += 0.3

    if action.priority == truth["priority"]:
        score += 0.3

    # smarter response check
    response = action.response.lower()

    if truth["label"] == "spam":
        if "ignore" in response or "spam" in response:
            score += 0.4
    else:
        if "reply" in response or "acknowledge" in response:
            score += 0.4

    return score