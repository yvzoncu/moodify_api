from playwright.sync_api import sync_playwright

def extract_track_urls_from_page(url: str) -> list:

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, timeout=60000)

        # Wait for the elements to load
        page.wait_for_selector("td.table_img a", timeout=10000)

        # Query all anchor tags within the specific td class
        track_elements = page.query_selector_all("td.table_img a")
        track_urls = [element.get_attribute("href") for element in track_elements if element.get_attribute("href")]

        browser.close()
        return track_urls
    

url = "https://songdata.io/charts/norway"
tracks = extract_track_urls_from_page(url)
print(tracks)