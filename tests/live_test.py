def test_classification_batch_array_live():
    """Simple live test: processes your array of texts through the REAL system (no mocks).
    Prints clean JSON classification for every text so you can see the live output."""
    from dotenv import load_dotenv
    load_dotenv()
    
    from messy_text.classifier import classify

    # ── Your array of texts ───────────────────────────────
    texts = [
    "Day 12 of principal photography. Cameras rolling on set in Toronto. Call sheet attached.",
    "Still stuck in dev hell – script rewrites ongoing + trying to attach director.",
    "Greenlit last week! Hired DP, scouting locations in Atlanta, shoot starts in 4 weeks.",
    "Post-production wrapping up, VFX shots look great. Trailer drops next month.",
    "cameras up rn day 3, crew exhausted but we got the shot lol call sheet 4 tmrw",
    "Just attached Zendaya to the lead. Still chasing financing with Netflix.",
    "Finally greenlit. Locked the budget, hired key crew, locations scouted, first AD starts Monday.",
    "Wrapped dev hell, green light came through yesterday, now PP almost done, locs locked, shooting next week.",
    "Big week on the project. Lots happening, more to come soon!",
    "S1 is in post, but the sequel is still in dev hell trying to attach the director again.",
    "Just secured the book rights. Pitch deck ready for studios next week.",
    "Final cut locked. Marketing team is running test screenings. Netflix wants it Q3.",
    "On set today. Rolling. Dailies look fire.",
    "Still rewriting pages but cameras roll in 10 days. Prep is crazy.",
    "Official poster dropped today. First look at the cast!",
    "   \n\t   ",
    "Talent attached + script polish complete, but no greenlight yet.",
    "PP wrapping up – art dept strike, final casting locked, shoot day 1 in 3 weeks.",
    "Dailies from yesterday look amazing. Wrap report for day 18 attached.",
    "No updates on the project. Just waiting on notes from the studio."
]

    print("\n=== LIVE BATCH CLASSIFICATION RESULTS (real Groq) ===\n")

    for i, text in enumerate(texts, 1):
        print(f"--- Text {i} ---")
        print(f"Input : {text}")
        result = classify(text) 
        print(result.model_dump_json(indent=2))
        print("-" * 60)