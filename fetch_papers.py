import arxiv
import os
import re

def clean_filename(title):
    """Removes special characters from the title to make it a valid filename."""
    # Keep only alphanumeric, spaces, underscores, and dashes
    clean = re.sub(r'[^\w\s-]', '', title)
    # Replace spaces with underscores and limit length
    return clean.replace(' ', '_')[:50]

def download_papers(topic, max_results=10):
    # 1. Create a data directory if it doesn't exist
    if not os.path.exists("data"):
        os.makedirs("data")

    # 2. Configure the search
    client = arxiv.Client()
    search = arxiv.Search(
        query = topic,
        max_results = max_results,
        sort_by = arxiv.SortCriterion.Relevance
    )

    print(f"🔎 Searching for top {max_results} papers on: '{topic}'...")
    
    results = list(client.results(search))
    
    if not results:
        print("❌ No papers found.")
        return

    # 3. Download each paper
    for i, paper in enumerate(results):
        try:
            # Create a safe filename
            safe_title = clean_filename(paper.title)
            filename = f"{safe_title}.pdf"
            path = os.path.join("data", filename)
            
            # Skip if already downloaded
            if os.path.exists(path):
                print(f"⚠️ Exists: {filename}")
                continue
                
            # Download
            paper.download_pdf(dirpath="data", filename=filename)
            print(f"✅ [{i+1}/{max_results}] Downloaded: {paper.title}")
            
        except Exception as e:
            print(f"❌ Error downloading {paper.title}: {e}")

    print(f"\n🎉 Done! Check the 'data' folder.")

# --- RUN IT ---
if __name__ == "__main__":
    user_topic = input("Enter a research topic to search (e.g. 'Graph Neural Networks'): ")
    download_papers(user_topic, max_results=10)
