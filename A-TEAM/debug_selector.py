import requests
from bs4 import BeautifulSoup

url = "https://nlrc.go.kr/nlrc/mainCase/judgment/index.do?pageIndex=1"
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, 'html.parser')

print("Title:", soup.title.get_text())

# Check for table rows
rows = soup.select('.bbs_list tbody tr')
print(f"Rows found with .bbs_list tbody tr: {len(rows)}")

if len(rows) == 0:
    # Print the first 1000 characters of the body to see what's there
    print("Body snippet:")
    print(soup.body.prettify()[:1000])

    # Try finding any table
    tables = soup.find_all('table')
    print(f"Total tables found: {len(tables)}")
    for i, table in enumerate(tables):
        print(f"Table {i} classes: {table.get('class')}")
