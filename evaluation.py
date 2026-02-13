def run_evaluation(rag):
    questions = [
        ("What is the refund period?", "fully"),
        ("How long does shipping take?", "fully"),
        ("Can I cancel after dispatch?", "fully"),
        ("Are international returns allowed?", "partial"),
        ("What is the late fee policy?", "partial"),
        ("Does the company offer free gym membership?", "none"),
        ("What is the CEO's salary?", "none"),
    ]

    print("\nEvaluation Results\n")

    for q, expected in questions:
        output = rag.answer_question(q)

        if expected == "none" and "not available" in output.lower():
            score = "✅ Correct"
        elif expected == "none":
            score = "❌ Hallucinated"
        else:
            score = "⚠️ Partial/Correct"

        print(f"Q: {q}")
        print(f"Expected: {expected}")
        print(f"Score: {score}\n")
        