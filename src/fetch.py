import requests
import pandas as pd
import json
import time
import os

BASE_URL = "https://www.metaculus.com/api2"

def fetch_resolved_binary_questions(max_questions=2000):
    questions = []
    url = f"{BASE_URL}/questions/"
    
    params = {
        "forecast_type": "binary",
        "resolved": "true",
        "format": "json",
        "limit": 100,
        "offset": 0
    }

    print("Fetching questions from Metaculus...")

    while len(questions) < max_questions:
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            break
        
        data = response.json()
        results = data.get("results", [])
        
        if not results:
            print("No more results.")
            break
        
        for q in results:
            # Get resolution from nested question object
            question = q.get("question", {})
            resolution = question.get("resolution")
            
            if resolution not in ["yes", "no"]:
                continue
            
            # Get community prediction from aggregations
            try:
                aggs = question.get("aggregations", {})
                prediction = None
                for key in ["unweighted", "recency_weighted", "single_aggregation"]:
                    try:
                        centers = aggs[key]["latest"]["centers"]
                        if centers:
                            prediction = centers[0]
                            break
                    except (KeyError, TypeError, IndexError):
                        continue
            except Exception:
                prediction = None

            # Get category
            category_list = q.get("projects", {}).get("category", [])
            category = category_list[0].get("name", "Uncategorized") if category_list else "Uncategorized"

            questions.append({
                "id": q.get("id"),
                "title": q.get("title"),
                "category": category,
                "resolution": 1 if resolution == "yes" else 0,
                "resolve_time": question.get("actual_resolve_time"),
                "community_prediction": prediction,
                "number_of_forecasters": q.get("nr_forecasters"),
                "created_time": q.get("created_at"),
            })
        
        print(f"Fetched {len(questions)} questions so far...")
        params["offset"] += 100
        time.sleep(0.5)
    
    return questions[:max_questions]


def save_raw_data(questions):
    os.makedirs("data/raw", exist_ok=True)
    
    with open("data/raw/questions.json", "w") as f:
        json.dump(questions, f, indent=2)
    
    df = pd.DataFrame(questions)
    df.to_csv("data/raw/questions.csv", index=False)
    
    print(f"Saved {len(questions)} questions to data/raw/")
    return df


if __name__ == "__main__":
    questions = fetch_resolved_binary_questions(max_questions=10000)
    df = save_raw_data(questions)
    print(df.head())
    print(f"\nTotal questions fetched: {len(df)}")
    print(f"Columns: {list(df.columns)}")